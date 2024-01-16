from typing import Tuple

import librosa
import torch
import numpy as np
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

import torchaudio.compliance.kaldi as ta_kaldi
from spafe.fbanks import gammatone_fbanks
class Cqt(torch.nn.Module):
    """Convert Raw audio to CQT

    
    """

    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 512,
        n_mels: int = 80,
        fmin: float = None,
        fmax: float = None,
        htk: bool = False,
        log_base: float = None,
    ):
        super().__init__()

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: CQTS
        # (B,T) -> (B,D,n_mels)
        def extract_cqt(wav):
            wav = np.squeeze(wav)
            cqt = librosa.cqt(wav, sr=16000,n_bins=80,hop_length=160)
            cqt = librosa.amplitude_to_db( np.abs(cqt)  )
            return cqt.T         
        output=[]
        output_lens=[]
        for i, instance in enumerate(feat):
            cqt = extract_cqt(feat[i].cpu().numpy())
            output.append(torch.Tensor(cqt))
            output_lens.append(cqt.shape[0])
        #Hardcode again
        logmel_feat = torch.stack(output,0).cuda()
        if ilens is not None:

            logmel_feat = logmel_feat.masked_fill(
                make_pad_mask(ilens, logmel_feat, 1), 0.0
            )
        else:
            output_lens = logmel_feat.new_full(
                [logmel_feat.size(0)], fill_value=logmel_feat.size(1), dtype=torch.long
            )
        return logmel_feat, output_lens.cuda()

#
#
class Mfcc(torch.nn.Module):
    """Convert Raw audio to MFCC

    
    """

    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 512,
        n_mels: int = 80,
        fmin: float = None,
        fmax: float = None,
        htk: bool = False,
        log_base: float = None,
    ):
        super().__init__()

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: CQTS
        # (B,T) -> (B,D,n_mels)
        def extract_mfcc(wav):
            mfcc = librosa.feature.mfcc(y=wav, sr=16000, n_mfcc=80,n_fft=400,hop_length=160)
            return mfcc.T
      
        output=[]
        output_lens=[]
        for i, instance in enumerate(feat):
            mfcc = extract_mfcc(feat[i].cpu().numpy())
            output.append(torch.Tensor(mfcc))
            output_lens.append(mfcc.shape[0])
        #Hardcode again
        logmel_feat = torch.stack(output,0).cuda()
        if ilens is not None:

            logmel_feat = logmel_feat.masked_fill(
                make_pad_mask(ilens, logmel_feat, 1), 0.0
            )
        else:
            output_lens = logmel_feat.new_full(
                [logmel_feat.size(0)], fill_value=logmel_feat.size(1), dtype=torch.long
            )
        return logmel_feat, output_lens.cuda()

#
#
class Gamma(torch.nn.Module):
    """Convert Raw audio to GAMMA

    
    """

    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 512,
        n_mels: int = 80,
        fmin: float = None,
        fmax: float = None,
        htk: bool = False,
        log_base: float = None,
    ):
        super().__init__()

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: CQTS
        # (B,T) -> (B,D,n_mels)
        def extract_gamma(wav):
            gammatone_filter_bank = gammatone_fbanks.gammatone_filter_banks(nfilts=80, nfft=400, fs=16000)
            y=librosa.util.normalize(wav)
            magnitude = np.abs(librosa.stft(y = y,win_length=400,n_fft=400,hop_length = 160))**2
            Gam=np.dot(gammatone_filter_bank[0],magnitude)
            LogGamSpec = librosa.power_to_db(Gam,ref=np.max)
            LogGamSpec = np.abs(LogGamSpec)+1e-9
            return LogGamSpec.T
      
        output=[]
        output_lens=[]
        for i, instance in enumerate(feat):
            gamma = extract_gamma(feat[i].cpu().numpy())
            output.append(torch.Tensor(gamma))
            output_lens.append(gamma.shape[0])
        #Hardcode again
        logmel_feat = torch.stack(output,0).cuda()
        if ilens is not None:

            logmel_feat = logmel_feat.masked_fill(
                make_pad_mask(ilens, logmel_feat, 1), 0.0
            )
        else:
            output_lens = logmel_feat.new_full(
                [logmel_feat.size(0)], fill_value=logmel_feat.size(1), dtype=torch.long
            )
        return logmel_feat, output_lens.cuda()

#
#
class GroupD(torch.nn.Module):
    """Convert Raw audio to GAMMA

    """

    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 512,
        n_mels: int = 80,
        fmin: float = None,
        fmax: float = None,
        htk: bool = False,
        log_base: float = None,
    ):
        super().__init__()

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: CQTS
        # (B,T) -> (B,D,n_mels)
        def extract_gd(wav): #<--their version
            stft = librosa.stft(wav, n_fft=400, hop_length=160, window='hamming')   
            phase = np.angle(stft)
            phase_diff = np.diff(phase)
            group_delay = -np.unwrap(phase_diff) / (2 * np.pi * 160)  # Convert radians to seconds
            mel_filterbank = librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)
            group_delay_mel = np.dot(mel_filterbank, group_delay)
            group_delay_mel = librosa.power_to_db(group_delay_mel, ref=np.max)
            return group_delay_mel.T
      
        output=[]
        output_lens=[]
        for i, instance in enumerate(feat):
            gd = extract_gd(feat[i].cpu().numpy())
            output.append(torch.Tensor(gd))
            output_lens.append(gd.shape[0])
        #Hardcode again
        logmel_feat = torch.stack(output,0).cuda()
        if ilens is not None:

            logmel_feat = logmel_feat.masked_fill(
                make_pad_mask(ilens, logmel_feat, 1), 0.0
            )
        else:
            output_lens = logmel_feat.new_full(
                [logmel_feat.size(0)], fill_value=logmel_feat.size(1), dtype=torch.long
            )
        return logmel_feat, output_lens.cuda()

class LogMel(torch.nn.Module):
    """Convert Raw audio to LOGMEL

    """

    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 512,
        n_mels: int = 80,
        fmin: float = None,
        fmax: float = None,
        htk: bool = False,
        log_base: float = None,
    ):
        super().__init__()

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: CQTS
        # (B,T) -> (B,D,n_mels)
        def extract_mel(wav):
            wav_tensor = torch.tensor(wav).unsqueeze(0)
            features = ta_kaldi.fbank(
                wav_tensor, num_mel_bins=80, sample_frequency=16000
            )
            return features.numpy()
      
        output=[]
        output_lens=[]
        for i, instance in enumerate(feat):
            mel = extract_mel(feat[i].cpu().numpy())
            output.append(torch.Tensor(mel))
            output_lens.append(mel.shape[0])
        #Hardcode again
        logmel_feat = torch.stack(output,0).cuda()
        if ilens is not None:

            logmel_feat = logmel_feat.masked_fill(
                make_pad_mask(ilens, logmel_feat, 1), 0.0
            )
        else:
            output_lens = logmel_feat.new_full(
                [logmel_feat.size(0)], fill_value=logmel_feat.size(1), dtype=torch.long
            )
        return logmel_feat, output_lens.cuda()
