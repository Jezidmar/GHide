import copy
from typing import Optional, Tuple, Union

import humanfriendly
import numpy as np
import torch
from typeguard import check_argument_types

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.log_mel import LogMel
from espnet2.layers.log_mel import Mfcc
from espnet2.layers.log_mel import Cqt
from espnet2.layers.log_mel import Gamma
from espnet2.layers.log_mel import GroupD

from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet.nets.pytorch_backend.frontends.frontend import Frontend


class DefaultFrontend(AbsFrontend):
    """Conventional frontend structure for ASR.

    # Hardcoded feature extractor for MEL,MFCC,CQT,GAMMA,GD features.
    """

    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: int = None,
        fmax: int = None,
        htk: bool = False,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        apply_stft: bool = True,
    ):
        assert check_argument_types()
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        # Deepcopy (In general, dict shouldn't be used as default arg)
        frontend_conf = copy.deepcopy(frontend_conf)
        self.hop_length = hop_length





        self.n_mels = n_mels
        self.frontend_type = "default"
        
        self.extract_feats = GroupD(            #<--Just replace LogMel with Gamma, Mfcc, Cqt, GroupD,LogMel
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
            )
        
        
    def output_size(self) -> int:
        return self.n_mels

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        #Input lengths are ;None; <-Hardcoded
        #Everything is hardcoded for now. Input is: 
        input_feats, feats_lens = self.extract_feats(input)
        return input_feats, feats_lens

