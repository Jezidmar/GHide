from typing import Tuple
import librosa
import torch
import numpy as np
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from nnAudio import Spectrogram
import torchaudio
import torchaudio.compliance.kaldi as ta_kaldi
import torchaudio.transforms as transforms
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
    
        self.extract_cqt = Spectrogram.CQT(sr=16000, n_bins=80, bins_per_octave=12, hop_length=160)

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: CQTS
        # (B,T) -> (B,D,n_mels)
#       
        output=[]
        output_lens=[]
        for i, instance in enumerate(feat):
            cqt = self.extract_cqt(feat[i].unsqueeze(0)).squeeze().T
            output.append(cqt)
            output_lens.append(cqt.shape[0])
        #Hardcode again
        cqt_feat = torch.stack(output,0)
        if ilens is not None:

            logmel_feat = cqt_feat.masked_fill(
                make_pad_mask(ilens, cqt_feat, 1), 0.0
            )
        else:
            output_lens = cqt_feat.new_full(
                [cqt_feat.size(0)], fill_value=cqt_feat.size(1), dtype=torch.long
            )
        return cqt_feat, output_lens.cuda()

#
#
class MfccLIBROSA(torch.nn.Module):
    """Convert Raw audio to MFCC
    #Mfcc
    
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
        logmel_feat = torch.stack(output,0)
        if ilens is not None:

            logmel_feat = logmel_feat.masked_fill(
                make_pad_mask(ilens, logmel_feat, 1), 0.0
            )
        else:
            output_lens = logmel_feat.new_full(
                [logmel_feat.size(0)], fill_value=logmel_feat.size(1), dtype=torch.long
            )
        return logmel_feat, output_lens


class Mfcc(torch.nn.Module):
    """Convert Raw audio to MFCC
    #Mfcc
    
    """

    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 400,
        n_mels: int = 80,
        fmin: float = None,
        fmax: float = None,
        htk: bool = False,
        log_base: float = None,
    ):
        super().__init__()
        self.fs = fs
        self.n_mels = n_mels
        self.mfcc_transform = transforms.MFCC(
            sample_rate=fs,
            n_mfcc=n_mels,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': 160,
                'n_mels': n_mels,
            }
        )

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: Raw audio waveform
        # (B, T) -> (B, D, n_mels)
        output = []
        output_lens = []
        
        for instance in feat:
            mfcc = self.mfcc_transform(instance)
            output.append(mfcc.T)
            output_lens.append(mfcc.size(1))
        
        logmel_feat = torch.stack(output, 0)
        
        if ilens is not None:
            logmel_feat = logmel_feat.masked_fill(
                make_pad_mask(ilens, logmel_feat, 1), 0.0
            )
        else:
            output_lens = logmel_feat.new_full(
                [logmel_feat.size(0)], fill_value=logmel_feat.size(1), dtype=torch.long
            )
        
        return logmel_feat, output_lens





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



class LogMelLIBROSA(torch.nn.Module):
    """Convert Raw audio to LOGMEL using librosa"""
    def __init__(self, fs: int = 16000, n_mels: int = 80, n_fft: int = 400, hop_length: int = 160, fmin: float = 0.0, fmax: float = None):
        super().__init__()
        self.fs = fs
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else fs // 2

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def extract_mel(wav):
            mel_spectrogram = librosa.feature.melspectrogram(
                y=wav, 
                sr=self.fs, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length, 
                n_mels=self.n_mels,
                #fmin=self.fmin,
                #fmax=self.fmax
            )
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
            return log_mel_spectrogram.T

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

class LogMel(torch.nn.Module):
    """Convert Raw audio to LOGMEL using torchaudio"""
    
    def __init__(self, fs: int = 16000, n_mels: int = 80, n_fft: int = 400, hop_length: int = 160, fmin: float = 0.0, fmax: float = None):
        super().__init__()
        self.fs = fs
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else fs // 2
        
        self.mel_spectrogram_transform = transforms.MelSpectrogram(
            sample_rate=fs,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        
        self.amplitude_to_db_transform = transforms.AmplitudeToDB()

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = []
        output_lens = []
        
        for instance in feat:
            mel_spectrogram = self.mel_spectrogram_transform(instance)
            log_mel_spectrogram = self.amplitude_to_db_transform(mel_spectrogram)
            output.append(log_mel_spectrogram.T)
            output_lens.append(log_mel_spectrogram.size(1))
        
        logmel_feat = torch.stack(output, 0).cuda()
        
        if ilens is not None:
            logmel_feat = logmel_feat.masked_fill(
                make_pad_mask(ilens, logmel_feat, 1), 0.0
            )
        else:
            output_lens = logmel_feat.new_full(
                [logmel_feat.size(0)], fill_value=logmel_feat.size(1), dtype=torch.long
            )
        
        return logmel_feat, torch.tensor(output_lens).cuda()

class LogMelDEF(torch.nn.Module):
    """Convert STFT to fbank feats

    The arguments is same as librosa.filters.mel

    Args:
        fs: number > 0 [scalar] sampling rate of the incoming signal
        n_fft: int > 0 [scalar] number of FFT components
        n_mels: int > 0 [scalar] number of Mel bands to generate
        fmin: float >= 0 [scalar] lowest frequency (in Hz)
        fmax: float >= 0 [scalar] highest frequency (in Hz).
            If `None`, use `fmax = fs / 2.0`
        htk: use HTK formula instead of Slaney
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

        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        _mel_options = dict(
            sr=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.mel_options = _mel_options
        self.log_base = log_base

        # Note(kamo): The mel matrix of librosa is different from kaldi.
        melmat = librosa.filters.mel(**_mel_options)
        # melmat: (D2, D1) -> (D1, D2)
        self.register_buffer("melmat", torch.from_numpy(melmat.T).float())

    def extra_repr(self):
        return ", ".join(f"{k}={v}" for k, v in self.mel_options.items())

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: (B, T, D1) x melmat: (D1, D2) -> mel_feat: (B, T, D2)
        mel_feat = torch.matmul(feat, self.melmat)
        mel_feat = torch.clamp(mel_feat, min=1e-10)

        if self.log_base is None:
            logmel_feat = mel_feat.log()
        elif self.log_base == 2.0:
            logmel_feat = mel_feat.log2()
        elif self.log_base == 10.0:
            logmel_feat = mel_feat.log10()
        else:
            logmel_feat = mel_feat.log() / torch.log(self.log_base)

        # Zero padding
        if ilens is not None:
            logmel_feat = logmel_feat.masked_fill(
                make_pad_mask(ilens, logmel_feat, 1), 0.0
            )
        else:
            ilens = feat.new_full(
                [feat.size(0)], fill_value=feat.size(1), dtype=torch.long
            )
        return logmel_feat, ilens


class LogMelTA(torch.nn.Module):
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
