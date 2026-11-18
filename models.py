"""PyTorch networks for the CIFAR-10 FAST configuration."""

import torch
from torch import nn
from torch.nn import functional as F


def conv_block(in_channels, out_channels, stride=1, *, transpose=False, activate=True):
    if transpose:
        conv = nn.ConvTranspose2d(in_channels, out_channels, 3, stride, padding=1,
                                  output_padding=stride - 1, bias=False)
    else:
        conv = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
    layers = [conv, nn.BatchNorm2d(out_channels)]
    if activate:
        layers.append(nn.PReLU())
    return nn.Sequential(*layers)


def normalize_features(features):
    """One unit of average power per complex symbol, independently per image."""
    power = 2 * features.square().mean(dim=(1, 2, 3), keepdim=True)
    return features * power.clamp_min(1e-12).rsqrt()


class Codec(nn.Module):
    channels = 24

    def __init__(self):
        super().__init__()
        self.encoder_layer1 = conv_block(3, 16)
        self.encoder_layer2 = conv_block(16, 32)
        self.encoder_layer3 = conv_block(32, 32, 2)
        self.encoder_layer4 = conv_block(32, 32, 2)
        self.encoder_layer5 = conv_block(32, self.channels, activate=False)
        self.decoder_layer1 = conv_block(self.channels, 32, transpose=True)
        self.decoder_layer2 = conv_block(32, 32, 2, transpose=True)
        self.decoder_layer3 = conv_block(32, 32, 2, transpose=True)
        self.decoder_layer4 = conv_block(32, 16, transpose=True)
        self.decoder_layer5 = conv_block(16, 3, transpose=True, activate=False)

    def encoder(self, images):
        x = self.encoder_layer1(images)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        return self.encoder_layer5(self.encoder_layer4(x))

    def decoder(self, features):
        x = self.decoder_layer1(features)
        x = self.decoder_layer2(x)
        x = self.decoder_layer3(x)
        return self.decoder_layer5(self.decoder_layer4(x))


@torch.no_grad()
def distillation_targets(codec, images):
    """Offline supervision for the released 2 x 2 evaluator."""
    features = codec.encoder(images)

    def reconstruction_psnr(maps):
        reconstruction = codec.decoder(normalize_features(maps)).clamp(0, 1)
        mse = (reconstruction - images).square().mean(dim=(1, 2, 3))
        return -10 * mse.clamp_min(1e-12).log10()

    reference = reconstruction_psnr(features)
    losses = []
    for k in range(features.shape[1]):
        masked = features.clone()
        masked[:, k] = 0
        losses.append(reference - reconstruction_psnr(masked))
    scores = torch.stack(losses, dim=1)
    low = scores.amin(dim=1, keepdim=True)
    span = scores.amax(dim=1, keepdim=True) - low
    safe_span = torch.where(span > 0, span, torch.ones_like(span))
    scores = torch.where(span > 0, (scores - low) / safe_span, 0.5)
    return features.detach(), scores.clamp_min(0.05)


class SEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels + 1, channels)
        self.fc2 = nn.Linear(channels, channels)
        self.ac1 = nn.PReLU()

    def forward(self, features, snr_db):
        pooled = features.mean(dim=(2, 3))
        snr = pooled.new_full((len(pooled), 1), snr_db)
        hidden = self.ac1(self.fc1(torch.cat((pooled, snr), dim=1)))
        weights = self.fc2(hidden).sigmoid()
        return features * weights[:, :, None, None]


class WNet(nn.Module):
    def __init__(self):
        super().__init__()
        channels = Codec.channels
        self.conv1 = conv_block(channels, channels)
        self.se1 = SEBlock(channels)
        self.conv2 = conv_block(channels, channels, 2)
        self.se2 = SEBlock(channels)
        self.conv3 = conv_block(channels, channels)
        self.fc1 = nn.Linear(4 * 4 * channels, channels)
        self.fc2 = nn.Linear(channels, channels)

    def forward(self, features, snr_db):
        x = self.se1(self.conv1(features), snr_db)
        x = self.se2(self.conv2(x), snr_db)
        x = self.conv3(x).flatten(1)
        return self.fc2(F.relu(self.fc1(x))).sigmoid()


class ChannelPredictor(nn.Module):
    """Predict the next complex 2 x 2 channel from 120 observed matrices."""
    history = 120
    refresh = 4
    antennas = 2

    def __init__(self):
        super().__init__()
        links = self.antennas ** 2
        self.lstm = nn.LSTM(2 * links, 120, num_layers=3, batch_first=True)
        self.fc1 = nn.Linear(120, links)
        self.fc2 = nn.Linear(120, links)

    def forward(self, history):
        sequence = torch.view_as_real(history).flatten(2)
        state_shape = (self.lstm.num_layers, len(sequence), self.lstm.hidden_size)
        # Match the random initial states used in training the released weights.
        h0 = sequence.new_empty(state_shape).normal_()
        c0 = sequence.new_empty(state_shape).normal_()
        output, _ = self.lstm(sequence, (h0, c0))
        last = output[:, -1]
        return torch.complex(self.fc1(last), self.fc2(last)).reshape_as(history[:, 0])

    def forecast(self, history, slots):
        predicted = []
        for _ in range(slots):
            next_csi = self(history)
            predicted.append(next_csi)
            history = torch.cat((history[:, 1:], next_csi[:, None]), dim=1)
        return torch.stack(predicted, dim=1)
