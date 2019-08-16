import logging

import os
from tqdm import tqdm


def file_lines_generator(filename, ignore_empty=False, skip=0, end=-1, binary=False):
    if end <= skip:
        end = os.path.getsize(filename)

    def _iterate():
        if binary:
            flag = 'rb'
            encoding = None
        else:
            flag = 'r'
            encoding = 'utf-8'
        with open(filename, flag, encoding=encoding) as f:
            f.seek(skip)
            while True:
                if f.tell() >= end:
                    break

                line = f.readline()
                if not line:
                    break
                if line.strip() or not ignore_empty:
                    yield line

    return _iterate()


def check_lines_of_file(filename, ignore_empty=False):
    i = 0
    with tqdm(total=os.path.getsize(filename)) as pbar:
        with open(filename, 'rb') as f:
            while True:
                line = f.readline()
                if not line:
                    break

                if line.strip() or not ignore_empty:
                    i += 1
                pbar.update(f.tell() - pbar.n)
    return i


def split_text_file(filename, output_filename_pattern, lines_per_block, ignore_empty=False):
    current_block_id = 0
    current_line_documents = 0
    current_filename = output_filename_pattern % current_block_id
    current_file = open(current_filename, 'wb')

    for line in file_lines_generator(filename, ignore_empty=ignore_empty, binary=True):
        current_file.write(line)
        current_line_documents += 1
        if current_line_documents >= lines_per_block:
            logging.info('Wrote %d lines to %s', current_line_documents, current_filename)
            current_file.close()
            current_block_id += 1
            current_line_documents = 0
            current_filename = output_filename_pattern % current_block_id
            current_file = open(current_filename, 'wb')

    if current_line_documents > 0:
        logging.info('Wrote %d lines to %s', current_line_documents, current_filename)
    current_file.close()
    if current_line_documents <= 0:
        os.unlink(current_filename)
        total_blocks = current_block_id
    else:
        total_blocks = current_block_id + 1

    return total_blocks


def divide_text_file(filename, lines_per_block, ignore_empty=False):
    current_line_documents = 0
    block_boundary_start = 0
    block_boundary_end = 0

    blocks = []

    with open(filename, 'rb') as f, tqdm(total=os.path.getsize(filename)) as pbar:
        while True:
            line = f.readline()
            if not line:
                break

            block_boundary_end = f.tell()
            if line.strip() or not ignore_empty:
                current_line_documents += 1

            if current_line_documents >= lines_per_block:
                blocks.append((block_boundary_start, block_boundary_end))
                current_line_documents = 0
                block_boundary_start = block_boundary_end

            pbar.update(f.tell() - pbar.n)

    if current_line_documents > 0:
        blocks.append((block_boundary_start, block_boundary_end))

    return blocks
