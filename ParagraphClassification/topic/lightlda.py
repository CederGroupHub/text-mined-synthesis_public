import collections
import logging
import os
import re
import subprocess
import sys

import numpy
from synthesis_project_ceder.nlp.token_filter import FilterClass

from ..nlp.preprocessing import TextPreprocessor
from ..nlp.token_storage import LabeledDocumentsReader
from ..nlp.vocabulary import Vocabulary
from ..utils.temp_file import TempDirname


class Filter(FilterClass):
    def __init__(self, *args, **kwargs):
        super(Filter, self).__init__(minimum_number_tokens=0)

        self.bad_symbols = re.compile(r'^SYMNUMBER|^GREEKCHAR|^LANGSYM_|^MATHCHAR_')

    def __call__(self, orth, lemma, pos):
        lst = super(Filter, self).__call__(orth, lemma, pos)
        if lst is None:
            return lst

        new_lst = [i for i in lst if not self.bad_symbols.match(i)]

        if len(new_lst) < self.minimum_number_tokens:
            return None
        return new_lst


class LightLDAOutput(object):
    def __init__(self, output_dir, input_dir=None):
        self.output_dir = output_dir
        self.input_dir = input_dir

        self._check_files()

    def _check_files(self):
        if len(list(self._find_summary_file())) == 0:
            raise ValueError('No topic summary table found.')

        if len(list(self._find_doc_topic_table_sorted())) == 0:
            logging.warning(
                'No doc topic files found at %s, you will not be able to read per-document topics.',
                self.output_dir
            )

        input_file = os.path.join(self.input_dir, 'corpus.libsvm')
        if not os.path.exists(input_file):
            logging.warning(
                'No input file found at %s, you will not be able to read per-document topics.',
                self.input_dir
            )

        if len(list(self._find_word_topic_file())) == 0:
            raise ValueError('No word topic table found.')

        self._find_dict_file()

        if len(list(self._find_log_file())) == 0:
            raise ValueError('No log file found.')

    @staticmethod
    def _parse_line(line):
        if isinstance(line, str):
            separator = r'\s+'
            field_separator = ':'
        elif isinstance(line, bytes):
            separator = br'\s+'
            field_separator = b':'
        else:
            raise TypeError('line must be a sequence.')

        fields = re.split(separator, line.strip())
        if len(fields) < 1:
            raise RuntimeError('No value found for line: ' + repr(line))

        line_label = fields[0]

        values = {}
        for i in fields[1:]:
            column, value = i.split(field_separator)
            try:
                values[int(column)] = int(value)
            except ValueError:
                raise RuntimeError('Cannot parse field into integers: ' + repr(i))

        return line_label, values

    @staticmethod
    def _parse_array_file(filename):
        array = {}
        with open(filename, 'rb') as f:
            for line in f:
                line_label, values = LightLDAOutput._parse_line(line)
                array[int(line_label)] = values

            return array

    def _find_summary_file(self):
        for i in os.listdir(self.output_dir):
            path = os.path.join(self.output_dir, i)
            if re.match(r'server_\d+_table_1\.model', i) and os.path.getsize(path) > 2:
                yield path

    def topic_summary_table(self):
        array = {}
        for summary_filename in self._find_summary_file():
            array.update(self._parse_array_file(summary_filename))

        if len(array) != 1 or 0 not in array:
            raise ValueError('Invalid summary array read.')
        return array[0]

    def _find_doc_topic_table_sorted(self):
        doc_topics_filenames = []
        for i in os.listdir(self.output_dir):
            m = re.match(r'doc_topic\.(\d+)\.(\d+)', i)
            if m is not None:
                major, minor = int(m.group(1)), int(m.group(2))
                doc_topics_filenames.append((major, minor, os.path.join(self.output_dir, i)))

        doc_topics_filenames = sorted(doc_topics_filenames)
        return [x[2] for x in doc_topics_filenames]

    def doc_topic_table(self):
        with LabeledDocumentsReader(
                os.path.join(self.input_dir, 'corpus.libsvm')) as doc_ids:
            doc_topic_tables = self._find_doc_topic_table_sorted()

            if len(doc_topic_tables) == 0:
                raise ValueError('No doc topic table found.')

            for filename in doc_topic_tables:
                with open(filename, 'rb') as f:
                    for line in f:
                        doc_label, _ = doc_ids.next()
                        _, values = self._parse_line(line)
                        yield doc_label, values

    def _find_word_topic_file(self):
        for i in os.listdir(self.output_dir):
            if re.match(r'server_\d+_table_0\.model', i):
                yield os.path.join(self.output_dir, i)

    def word_topic_table(self):
        topic_words = {}
        for filename in self._find_word_topic_file():
            topic_words.update(self._parse_array_file(filename))

        return topic_words

    def word_topic_table_sorted(self):
        topic_words = self.word_topic_table()
        for t, v in topic_words.items():
            s = sum(v.values())
            v = sorted([(y / s, x) for x, y in v.items()], reverse=True)
            topic_words[t] = v
        return topic_words

    def _find_dict_file(self):
        for i in os.listdir(self.input_dir):
            if re.match(r'.*?\.dict$', i):
                return os.path.join(self.input_dir, i)

        raise RuntimeError('No dict found!')

    def dictionary(self):
        with open(self._find_dict_file()) as f:
            dict_ids = {}
            for line in f:
                word_id_s, word_s, count_s = re.split(r'\s+', line.strip())
                word_id, count = int(word_id_s), int(count_s)

                dict_ids[word_id] = (word_s, count)

            return dict_ids

    def _find_log_file(self):
        for i in os.listdir(self.output_dir):
            m = re.match(r'LightLDA\.(\d+)\.\d+\.log', i)
            if m is None:
                continue
            yield int(m.group(1)), os.path.join(self.output_dir, i)

    def model_likelihood(self):
        blocks = {}
        for blockid, logfile in self._find_log_file():
            this_iter = None
            doc_likelihood = []
            word_likelihood = []
            normalized_likelihood = []

            with open(logfile) as f:
                for line in f:
                    m = re.match(r'.*?Iter = (\d+).*', line)
                    if m:
                        this_iter = int(m.group(1))
                        continue

                    m = re.match(r'.*?doc likelihood : (.*)', line)
                    if m:
                        doc_likelihood.append((float(m.group(1)), this_iter))
                        continue

                    m = re.match(r'.*?word likelihood : (.*)', line)
                    if m:
                        word_likelihood.append((float(m.group(1)), this_iter))
                        continue

                    m = re.match(r'.*?Normalized likelihood : (.*)', line)
                    if m:
                        normalized_likelihood.append((float(m.group(1)), this_iter))
                        continue

            y, x = zip(*doc_likelihood)
            doc_likelihood = numpy.array(x), numpy.array(y)

            y, x = zip(*word_likelihood)
            word_likelihood = numpy.array(x), numpy.array(y)

            y, x = zip(*normalized_likelihood)
            normalized_likelihood = numpy.array(x), numpy.array(y)
            assert numpy.equal(doc_likelihood[0], word_likelihood[0]).all()
            assert numpy.equal(doc_likelihood[0], normalized_likelihood[0]).all()

            word_likelihood = word_likelihood[0], word_likelihood[1] + normalized_likelihood[1]

            blocks[blockid] = {
                'doc': doc_likelihood,
                'word': word_likelihood
            }
        return blocks


class LightLDAInference(LightLDAOutput):
    LINE_REGEX = re.compile(r'Topics for \d+: ((?:\d+:\d+\s+)*)\s*$')

    def __init__(self, lightlda_inference, ntopics, nvocabulary, alpha,
                 beta=0.01, niterations=200, mh_steps=10, random_seed=-1,
                 *args, **kwargs):
        super(LightLDAInference, self).__init__(*args, **kwargs)
        self.vocabulary = Vocabulary.load(os.path.join(self.input_dir, 'corpus.dict'))

        self.command_line = [
            os.path.realpath(lightlda_inference),
            '-rand', '%d' % random_seed,
            '-num_vocabs', '%d' % nvocabulary,
            '-num_topics', '%d' % ntopics,
            '-num_iterations', '%d' % niterations,
            '-mh_steps', '%d' % mh_steps,
            '-alpha', '%f' % alpha,
            '-beta', '%f' % beta,
            '-max_num_document', '1',
            '-input_dir', os.path.realpath(self.output_dir)
        ]
        self.tmp_dir = TempDirname()
        self.tmp_dir_path = None
        self.pipe = None

    def __enter__(self):
        self.tmp_dir_path = self.tmp_dir.__enter__()
        logging.debug('Executing lightlda as %r', self.command_line)
        self.pipe = subprocess.Popen(
            args=self.command_line,
            cwd=os.path.realpath(self.tmp_dir_path),
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            encoding='utf8'
        )
        self.pipe.__enter__()

        # This is actually a hack. Make sure the folder is removed.
        self.pipe.stdout.readline()
        self.tmp_dir.__exit__(*sys.exc_info())

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pipe.__exit__(exc_type, exc_val, exc_tb)

    def _process_text(self, text):
        processor = TextPreprocessor(text)
        all_lemmas = processor.get_words(lemma=True)
        all_orths = processor.get_words(lemma=False)
        all_pos = processor.get_pos()
        # all_orths, all_pos = [], []

        # for sentence in processor.doc.user_data.sentences:
        #     orths, pos = zip(*sentence.pos_tagged_tokens)
        #     for x in orths: all_orths.append(x)
        #     for x in pos: all_pos.append(x)

        token_filter = Filter(minimum_number_tokens=0)
        tokens = token_filter(all_orths, all_lemmas, all_pos)
        if tokens is None:
            tokens = []

        tokensid = collections.defaultdict(int)
        for i in tokens:
            if i in self.vocabulary.word2id:
                _id = self.vocabulary.word2id[i]
                tokensid[_id] += 1

        if len(tokensid) == 0:
            return None

        input_line = []
        for tokenid, count in tokensid.items():
            input_line.append('%d:%d' % (tokenid, count))
        input_line.append('\n')
        input_line = ' '.join(input_line)

        return input_line

    def _read_one_topic_line(self):
        while True:
            line = self.pipe.stdout.readline()
            logging.debug('lightlda inference returned: %s', line.strip())

            if not line:
                logging.error('lightlda did not give expected output string.')
                raise ValueError('lightlda did not give expected output string.')

            m = self.LINE_REGEX.search(line)
            if m:
                topics_line = m.group(1).strip()
                topics = {}

                if topics_line:
                    for i in re.split(r'\s+', topics_line):
                        t, c = i.split(':')
                        t, c = int(t), int(c)
                        topics[t] = c

                return topics

    def _feed_text_buffered(self, list_of_text, buffer_size=1024):
        """
        Note: list_of_text will be modified.
        """
        while len(list_of_text) > 0:
            text_written = []
            written_bytes = 0

            while written_bytes < buffer_size and len(list_of_text):
                text = list_of_text.pop(0)

                if text is None:
                    text_written.append(False)
                else:
                    self.pipe.stdin.write(text)
                    written_bytes += len(text)
                    text_written.append(True)

            self.pipe.stdin.flush()
            for written in text_written:
                yield written

    def _infer(self, list_of_text, repeat):
        input_lines = []
        for text in list_of_text:
            this_line = self._process_text(text)
            for i in range(repeat):
                input_lines.append(this_line)

        for written in self._feed_text_buffered(input_lines):
            if not written:
                yield {}
            else:
                yield self._read_one_topic_line()

    def infer(self, texts, repeat=1):
        status = self.pipe.poll()
        if status is not None:
            raise RuntimeError('lightlda has terminated with error code: %d' % status)

        topic_results = [t for t in self._infer(texts, repeat)]

        return topic_results
