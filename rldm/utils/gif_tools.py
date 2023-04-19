import base64
import glob
import io
import json
import os
import subprocess

import numpy as np
from gfootball.env.script_helpers import ScriptHelpers as sh


def get_gif_html(videos_path, title):
    videos = np.array(glob.glob(videos_path))
    n_videos = len(videos)
    if n_videos == 0:
        return '<h1>No "{}" videos found.<h1>'.format(title)

    vids = {}
    for idx, video in enumerate(videos):
        basename = os.path.splitext(video)[0]
        gif_path = basename + '.gif'
        dump_path = basename + '.dump'
        ps = subprocess.Popen(
            ('ffmpeg', 
                '-hide_banner', '-nostats',
                '-loglevel', 'panic',
                '-i', video, 
                '-r', '10',
                '-f', 'image2pipe', 
                '-vcodec', 'ppm',
                '-crf', '20',
                '-vf', 'scale=300:-1',
                '-'), 
            stdout=subprocess.PIPE)
        output = subprocess.check_output(
            ('convert',
                '-quiet',
                '-coalesce',
                '-delay', '7',
                '-loop', '0',
                '-fuzz', '2%',
                '+dither',
                '-deconstruct',
                '-layers', 'Optimize',
                '-', gif_path), 
            stdin=ps.stdout)
        ps.wait()

        gif = io.open(gif_path, 'r+b').read()
        encoded = base64.b64encode(gif)

        html_tag = """
        <h3>{0}<h3/>
        <img src="data:image/gif;base64,{1}" width="800"/>"""
        sub = ''
        d = sh().load_dump(dump_path)
        episode_id = d[0]['debug']['config']['episode_number'] - 1
        sub = f'Episode {episode_id}'
        vids[episode_id] = html_tag.format(sub, encoded.decode('ascii'))

    strm = '<h1>{}, {} videos:<h1>'.format(title, n_videos)
    for _, v in sorted(vids.items()):
        strm += v
    return strm

def get_gif_html_oai(env_videos, title, subtitle_eps=None, max_n_videos=4):
    videos = np.array(env_videos)
    if len(videos) == 0:
        return
    
    n_videos = max(1, min(max_n_videos, len(videos)))
    idxs = np.linspace(0, len(videos) - 1, n_videos).astype(int) if n_videos > 1 else [-1,]
    videos = videos[idxs,...]

    strm = '<h2>{}<h2>'.format(title)
    for video_path, meta_path in videos:
        basename = os.path.splitext(video_path)[0]
        gif_path = basename + '.gif'
        if not os.path.exists(gif_path):
            ps = subprocess.Popen(
                ('ffmpeg', 
                 '-i', video_path, 
                 '-r', '7',
                 '-f', 'image2pipe', 
                 '-vcodec', 'ppm',
                 '-crf', '20',
                 '-vf', 'scale=512:-1',
                 '-'), 
                stdout=subprocess.PIPE)
            output = subprocess.check_output(
                ('convert',
                 '-coalesce',
                 '-delay', '7',
                 '-loop', '0',
                 '-fuzz', '2%',
                 '+dither',
                 '-deconstruct',
                 '-layers', 'Optimize',
                 '-', gif_path), 
                stdin=ps.stdout)
            ps.wait()

        gif = io.open(gif_path, 'r+b').read()
        encoded = base64.b64encode(gif)
            
        with open(meta_path) as data_file:    
            meta = json.load(data_file)

        html_tag = """
        <h3>{0}<h3/>
        <img src="data:image/gif;base64,{1}" />"""
        prefix = 'Trial ' if subtitle_eps is None else 'Episode '
        sufix = str(meta['episode_id'] if subtitle_eps is None \
                    else subtitle_eps[meta['episode_id']])
        strm += html_tag.format(prefix + sufix, encoded.decode('ascii'))
    return strm
