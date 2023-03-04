import logging
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy
import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write

logging.basicConfig(level=logging.WARNING)


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


hps_mt = utils.get_hparams_from_file("./configs/genshin.json")

npcList = [
    '派蒙', '凯亚', '安柏', '丽莎', '琴', '香菱', '枫原万叶', '迪卢克', '温迪', '可莉', '早柚', '托马',
    '芭芭拉', '优菈', '云堇', '钟离', '魈', '凝光', '雷电将军', '北斗', '甘雨', '七七', '刻晴', '神里绫华',
    '戴因斯雷布', '雷泽', '神里绫人', '罗莎莉亚', '阿贝多', '八重神子', '宵宫', '荒泷一斗', '九条裟罗', '夜兰',
    '珊瑚宫心海', '五郎', '散兵', '女士', '达达利亚', '莫娜', '班尼特', '申鹤', '行秋', '烟绯', '久岐忍',
    '辛焱', '砂糖', '胡桃', '重云', '菲谢尔', '诺艾尔', '迪奥娜', '鹿野院平藏'
]

net_g_mt = SynthesizerTrn(len(symbols),
                          hps_mt.data.filter_length // 2 + 1,
                          hps_mt.train.segment_size // hps_mt.data.hop_length,
                          n_speakers=hps_mt.data.n_speakers,
                          **hps_mt.model).cuda()
_ = net_g_mt.eval()

_ = utils.load_checkpoint("./models/G_809000.pth", net_g_mt, None)


def speak(sentence: str, speaker: str = "钟离"):
    audio_mt = numpy.array([])

    stn_tst_mt = get_text(
        sentence.replace("\n", "").replace('\u3000', ''), hps_mt)
    with torch.no_grad():
        x_tst_mt = stn_tst_mt.cuda().unsqueeze(0)
        x_tst_mt_lengths = torch.LongTensor([stn_tst_mt.size(0)]).cuda()
        sid_mt = torch.LongTensor([npcList.index(speaker)]).cuda()
        audio_mt = net_g_mt.infer(
            x_tst_mt,
            x_tst_mt_lengths,
            sid=sid_mt,
            noise_scale=.4,
            noise_scale_w=.8,
            length_scale=1.2)[0][0, 0].data.cpu().float().numpy()
        return audio_mt


if __name__ == "__main__":
    audio_mt = speak("随着文本转语音技术的发展，当前的踢踢挨斯技术已经强大到可以模仿游戏人物的语音了！好神奇！", '神里绫华')
    write(filename="a.wav", rate=hps_mt.data.sampling_rate, data=audio_mt)
