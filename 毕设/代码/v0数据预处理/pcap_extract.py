import re
import numpy as np
import os
import collections
import json
import sys
import subprocess
import shutil
import logging

# tshark_path = r"E:\Program Files\Wireshark\tshark.exe"
# editcap_path = r"E:\Program Files\Wireshark\editcap.exe"
tshark_path = r"D:\Program Files\Wireshark\tshark.exe"
editcap_path = r"D:\Program Files\Wireshark\editcap.exe"

SAVE_FOLDER = "output"
_TWO_20 = float(2 ** 20)  # 1M
SIZE_THRESHOLD = 10 * _TWO_20  # 10M
PACKETS_PER_FILE = 10000


def get_streams(json_folder, logger, save_folder=SAVE_FOLDER):
    flow_dict = collections.defaultdict(list)

    for file in os.listdir(json_folder):
        frames_text = os.path.join(json_folder, file)
        try:
            with open(frames_text, mode='r', encoding='UTF-8', errors='ignore') as f:
                frames_json = json.load(f)
                for frame in frames_json:
                    frame_protocols = frame['_source']['layers']['frame']['frame.protocols']
                    frame_number = frame['_source']['layers']['frame']['frame.number']
                    try:
                        if 'ip' in frame_protocols and ('tcp' in frame_protocols or 'udp' in frame_protocols):
                            ip_key = frame_protocols.split(":")[2]
                            ip_src = frame['_source']['layers'][ip_key][ip_key + '.src']
                            ip_dst = frame['_source']['layers'][ip_key][ip_key + '.dst']
                            if ip_key == 'ip':
                                ip_proto = frame['_source']['layers'][ip_key][ip_key + '.proto']
                            else:
                                ip_proto = frame['_source']['layers'][ip_key][ip_key + '.nxt']
                            if ip_proto == '6':
                                tcp_packet = frame['_source']['layers']['tcp']
                                if 'tcp.analysis' in tcp_packet:
                                    if 'tcp.analysis.flags' in tcp_packet['tcp.analysis']:
                                        if '_ws.expert' in tcp_packet['tcp.analysis']['tcp.analysis.flags']:
                                            if 'tcp.analysis.retransmission' in \
                                                    tcp_packet['tcp.analysis']['tcp.analysis.flags']['_ws.expert'] or \
                                                    'tcp.analysis.out_of_order' in \
                                                    tcp_packet['tcp.analysis']['tcp.analysis.flags']['_ws.expert']:
                                                continue
                                srcport = frame['_source']['layers']['tcp']['tcp.srcport']
                                dstport = frame['_source']['layers']['tcp']['tcp.dstport']
                            if ip_proto == '17':
                                srcport = frame['_source']['layers']['udp']['udp.srcport']
                                dstport = frame['_source']['layers']['udp']['udp.dstport']
                            stream = frame['_source']['layers']['frame_raw'][0]
                            if ip_key == 'ip':
                                flow_key = "-".join(sorted([ip_src.replace('.', '_') + '-' + srcport,
                                                            ip_dst.replace('.', '_') + '-' + dstport])) + '-' + ip_proto
                            else:
                                flow_key = "-".join(sorted([ip_src.replace(':', '_') + '-' + srcport,
                                                            ip_dst.replace(':', '_') + '-' + dstport])) + '-' + ip_proto
                            flow_dict[flow_key].append(
                                [ip_src, srcport, ip_dst, dstport, ip_proto, stream, frame_number])
                    except Exception as e:
                        logger.error(f"{e}, {file}, {frame}")
        except Exception as e:
            logger.error(f"{e}, {file}")

    print(f"read {json_folder} success!")
    logger.info(f"read {json_folder} success!")

    result_count = 1
    result_key = []
    for idx, (key, value) in enumerate(flow_dict.items()):
        if len(value) < 3:
            continue
        data = np.zeros((32, 512))
        for i in range(min(32, len(value))):
            hex_list = re.findall(r'.{2}', value[i][5])
            for j in range(min(len(hex_list), 512)):
                data[i][j] = int(hex_list[j], 16)
        data.resize((128, 128))
        sava_path = os.path.join(save_folder, str(result_count) + '.txt')
        np.savetxt(sava_path, data, fmt='%d')
        result_count += 1
        result_key.append(value[0][:5])
        print("save", sava_path, '\t', ' '.join(result_key[-1]))
        logger.info(f"save {sava_path} {' '.join(result_key[-1])}")

    with open(os.path.join(save_folder, 'info.txt'), 'w') as f:
        for info in result_key:
            f.write(' '.join(info) + '\n')

    print("Extract finish!")


def split_pcap(capture_path, logger, save_folder=SAVE_FOLDER):
    cmd = [editcap_path, '-c', str(PACKETS_PER_FILE), '-F', 'pcap', capture_path, os.path.join(save_folder, "tmp")]
    res = subprocess.run(cmd)

    if res.returncode == 0:
        logger.info(f"split {capture_path} success!")
        return 0
    else:
        logger.error(f"split {capture_path} error! {res.returncode} {res.stderr}")
        return res.returncode


def extract_json(capture_path, logger, save_folder=SAVE_FOLDER):
    file = os.path.splitext(os.path.basename(capture_path))[0]
    json_path = os.path.join(save_folder, file + '.json')

    cmd = [tshark_path, '-r', capture_path, '-j', r'" frame ip ipv6 tcp udp "', '-x', '-T', 'json']
    res = subprocess.run(cmd, stdout=subprocess.PIPE, encoding='utf-8')

    if res.returncode == 0:
        with open(json_path, mode='w', encoding='utf-8') as f:
            f.write(res.stdout)
        logger.info(f"{capture_path} json success!")
        return 0

    else:
        logger.error(f"{capture_path} json success! {res.returncode} {res.stderr}")
        return res.returncode


def main(capture_path, save_folder=SAVE_FOLDER, remove=True):
    if os.path.exists(save_folder):
        try:
            shutil.rmtree(save_folder)
        except Exception as e:
            print("Warning: Remove error", e)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    logfile = os.path.join(save_folder, 'logging.log')
    fmt = "%(asctime)s %(funcName)s:%(lineno)d:%(levelname)s: %(message)s"
    dfmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(filename=logfile, level=logging.INFO, format=fmt, datefmt=dfmt)
    logger = logging.getLogger()

    json_folder = os.path.join(save_folder, 'extract_json')
    if os.path.exists(json_folder):
        shutil.rmtree(json_folder)
    os.mkdir(json_folder)

    split_folder = os.path.join(save_folder, 'split_pcap')
    if os.path.exists(split_folder):
        shutil.rmtree(split_folder)
    os.mkdir(split_folder)

    result_folder = os.path.join(save_folder, 'result')
    if os.path.exists(result_folder):
        shutil.rmtree(result_folder)
    os.mkdir(result_folder)

    pcap_size = os.path.getsize(capture_path)
    logger.info(f"The size of {capture_path} is {pcap_size / _TWO_20 :.0f}M")
    if pcap_size < (SIZE_THRESHOLD * 2):
        res = extract_json(capture_path, logger, json_folder)
        if res != 0:
            raise Exception(f"extrct {capture_path} error! {res}")
    else:
        logger.info(f"start split {capture_path}")

        res = split_pcap(capture_path, logger, save_folder=split_folder)
        if res != 0:
            raise Exception(f"split {capture_path} error! {res}")

        for file in os.listdir(split_folder):
            res = extract_json(os.path.join(split_folder, file), logger, json_folder)
            if res != 0:
                raise Exception(f"extrct {capture_path} error! {res}")

    print(f"{json_folder} json success!")

    get_streams(json_folder, logger, result_folder)

    if remove:
        shutil.rmtree(split_folder)
        shutil.rmtree(json_folder)
        logger.info(f"remove {split_folder} and {json_folder}!")


if __name__ == "__main__":

    #D:\PychramProject\20230105\Siamese-pytorch-master\Siamese-pytorch-master\data_pcap\normal
    file = 'data_pcap\\normal\\kugou.pcap'

    # if len(sys.argv) < 2:
    #     raise Exception("Please input capture file path!")
    #
    # file = sys.argv[1]

    main(file)  # , remove=False
