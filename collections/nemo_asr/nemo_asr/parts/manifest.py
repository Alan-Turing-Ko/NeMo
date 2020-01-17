# Taken straight from Patter https://github.com/ryanleary/patter
# TODO: review, and copyright and fix/add comments
# TODO(stanislavv): Add logging for parsing errors.
import collections
import json
import string
from typing import Optional, List

import frozendict
import pandas as pd
from nemo_asr.parts import cleaners


class CharParser:
    def __init__(
        self,
        labels: List[str],
        unk_id: int = -1,
        blank_id: int = -1,
        do_normalize: bool = True,
        do_lowercase: bool = True,
    ):
        self.labels = labels
        self.unk_id = unk_id
        self.blank_id = blank_id
        self.do_normalize = do_normalize
        self.do_lowercase = do_lowercase

        self._labels_map = {label: index for index, label in enumerate(labels)}
        self._special_labels = set(
            [label for label in labels if len(label) > 1]
        )

    def __call__(self, text: str) -> Optional[List[int]]:
        if self.do_normalize:
            text = self._normalize(text)
            if text is None:
                return None

        text_tokens = self._tokenize(text)

        return text_tokens

    def _normalize(self, text: str) -> Optional[str]:
        text = text.strip()

        if self.do_lowercase:
            text = text.lower()

        return text

    def _tokenize(self, text):
        tokens = []
        # Split by word for find special labels.
        for word_id, word in enumerate(text.split(' ')):
            if word_id != 0:  # Not first word - so we insert space before.
                tokens.append(self._labels_map.get(' ', self.unk_id))

            if word in self._special_labels:
                tokens.append(self._labels_map[word])
                continue

            for char in word:
                tokens.append(self._labels_map.get(char, self.unk_id))

        # If unk_id == blank_id, OOV tokens are removed.
        tokens = [token for token in tokens if token != self.blank_id]

        return tokens


class ENCharParser(CharParser):
    PUNCTUATION_TO_REPLACE = frozendict.frozendict(
        {'+': 'plus', '&': 'and', '%': 'percent'}
    )

    def __init__(
        self, labels: List[str], unk_id: int = -1, blank_id: int = -1,
    ):
        super().__init__(labels, unk_id, blank_id)

        self._table = self.__make_trans_table()

    def __make_trans_table(self):
        punctuation = string.punctuation

        for char in self.PUNCTUATION_TO_REPLACE:
            punctuation = punctuation.replace(char, '')

        for label in self.labels:
            punctuation = punctuation.replace(label, '')

        table = str.maketrans(punctuation, ' ' * len(punctuation))

        return table

    def _normalize(self, text: str) -> Optional[str]:
        # noinspection PyBroadException
        try:
            text = cleaners.clean_text(
                string=text,
                table=self._table,
                punctuation_to_replace=self.PUNCTUATION_TO_REPLACE,
            )
        except Exception:
            return None

        return text


NAME_TO_PARSER = frozendict.frozendict(
    {'base': CharParser, 'en': ENCharParser}
)


def make_parser(labels, name='base', **kwargs):
    return NAME_TO_PARSER[name](labels, **kwargs)


def __load_raw(file):
    with open(file, 'r') as f:
        return f.read().replace('\n', '')


def __asr_json_gen(json_file):
    with open(json_file, 'r') as f:
        for line in f:
            yield json.loads(line)


Manifest = collections.UserList


class TextManifest(Manifest):
    OUTPUT_TYPE = collections.namedtuple('TextEntity', 'tokens')

    def __init__(self, texts: List[str], parser: CharParser):
        data = [self.OUTPUT_TYPE(parser(text)) for text in texts]
        super().__init__(data)


class FromFileTextManifest(TextManifest):
    def __init__(self, file: str, parser: CharParser):
        texts = self.__parse_texts(file)
        super().__init__(texts, parser)

    @staticmethod
    def __parse_texts(file: str) -> List[str]:
        if not os.path.exists(file):
            raise ValueError('Provided texts file does not exists!')

        _, ext = os.path.splitext(file)
        if ext == '.csv':
            texts = pd.read_csv(file)['transcript'].tolist()
        elif ext == '.json':
            texts = []
            for item in __asr_json_gen(file):
                if 'text' in item:
                    text = item['text']
                elif 'text_filepath' in item:
                    text = __load_raw(item['text_filepath'])
                else:
                    raise ValueError('Invalid json structure.')

                texts.append(text)
        else:
            with open(file, 'r') as f:
                texts = f.readlines()

        return texts


class AudioTextManifest(Manifest):
    def __init__(
        self,
        audio_files: List[str],
        durations: List[float],
        texts: List[str],
        parser: CharParser,
    ):
        # # NOT lazy loading.
        #
        # if parser not in NAME_TO_PARSER:
        #     raise ValueError('Invalid parser name.')
        #
        # parser_class = NAME_TO_PARSER[parser]
        # self.parser = parser_class(
        #     labels=labels,
        #     unk_id=unk_id,
        #     blank_id=blank_id,
        #     do_normalize=do_normalize,
        # )

        data = []  # TODO: ...
        super().__init__(data)


class ASRMAnifest(Manifest):
    def __init__(self, manifest_paths, labels, parser='base'):
        # parsing ...
        audio_file, text, duration = parse()
        super().__init__(audio_file, text, duration, parser=parser)


class ManifestBase:
    def __init__(
        self,
        manifest_paths,
        labels,
        max_duration=None,
        min_duration=None,
        sort_by_duration=False,
        max_utts=0,
        blank_index=-1,
        unk_index=-1,
        normalize=True,
        logger=None,
    ):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.sort_by_duration = sort_by_duration
        self.max_utts = max_utts
        self.blank_index = blank_index
        self.unk_index = unk_index
        self.normalize = normalize
        self.labels_map = {label: i for i, label in enumerate(labels)}
        self.logger = logger
        if logger is None:
            self.logger = get_logger("")

        data = []
        duration = 0.0
        filtered_duration = 0.0

        for item in self.json_item_gen(manifest_paths):
            if min_duration and item["duration"] < min_duration:
                filtered_duration += item["duration"]
                continue
            if max_duration and item['duration'] > max_duration:
                filtered_duration += item['duration']
                continue

            # load and normalize transcript text, i.e. `text`
            text = ""
            if "text" in item:
                text = item["text"]
            elif "text_filepath" in item:
                text = self.load_transcript(item["text_filepath"])
            else:
                filtered_duration += item["duration"]
                continue
            if normalize:
                text = self.normalize_text(text, labels, logger=self.logger)
            if not isinstance(text, str):
                self.logger.warning(
                    "WARNING: Got transcript: {}. It is not a "
                    "string. Dropping data point".format(text)
                )
                filtered_duration += item["duration"]
                continue
            # item['text'] = text

            # tokenize transcript text
            item["tokens"] = self.tokenize_transcript(
                text, self.labels_map, self.unk_index, self.blank_index
            )

            # support files using audio_filename
            if "audio_filename" in item and "audio_filepath" not in item:
                self.logger.warning(
                    "Malformed manifest: The key audio_filepath was not "
                    "found in the manifest. Using audio_filename instead."
                )
                item["audio_filepath"] = item["audio_filename"]

            data.append(item)
            duration += item["duration"]

            if max_utts > 0 and len(data) >= max_utts:
                self.logger.info(
                    "Stop parsing due to max_utts ({})".format(max_utts)
                )
                break

        if sort_by_duration:
            data = sorted(data, key=lambda x: x["duration"])
        self._data = data
        self._size = len(data)
        self._duration = duration
        self._filtered_duration = filtered_duration

    @staticmethod
    def normalize_text(text, labels):
        """for the base class remove surrounding whitespace only"""
        return text.strip()

    @staticmethod
    def tokenize_transcript(transcript, labels_map, unk_index, blank_index):
        """tokenize transcript to convert words/characters to indices"""
        # allow for special labels such as "<NOISE>"
        special_labels = set([l for l in labels_map.keys() if len(l) > 1])
        tokens = []
        # split by word to find special tokens
        for i, word in enumerate(transcript.split(" ")):
            if i > 0:
                tokens.append(labels_map.get(" ", unk_index))
            if word in special_labels:
                tokens.append(labels_map.get(word))
                continue
            # split by character to get the rest of the tokens
            for char in word:
                tokens.append(labels_map.get(char, unk_index))
        # if unk_index == blank_index, OOV tokens are removed from transcript
        tokens = [x for x in tokens if x != blank_index]
        return tokens

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return self._size

    def __iter__(self):
        return iter(self._data)

    @staticmethod
    def json_item_gen(manifest_paths):
        for manifest_path in manifest_paths:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    yield json.loads(line)

    @staticmethod
    def load_transcript(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as transcript_file:
            transcript = transcript_file.read().replace("\n", "")
        return transcript

    @property
    def duration(self):
        return self._duration

    @property
    def filtered_duration(self):
        return self._filtered_duration

    @property
    def data(self):
        return list(self._data)


class ManifestEN(ManifestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def normalize_text(text, labels, logger=None):
        # Punctuation to remove
        punctuation = string.punctuation
        # Define punctuation that will be handled by text cleaner
        punctuation_to_replace = {"+": "plus", "&": "and", "%": "percent"}
        for char in punctuation_to_replace:
            punctuation = punctuation.replace(char, "")
        # We might also want to consider:
        # @ -> at
        # -> number, pound, hashtag
        # ~ -> tilde
        # _ -> underscore

        # If a punctuation symbol is inside our vocab, we do not remove
        # from text
        for l in labels:
            punctuation = punctuation.replace(l, "")

        # Turn all other punctuation to whitespace
        table = str.maketrans(punctuation, " " * len(punctuation))

        try:
            text = clean_text(text, table, punctuation_to_replace)
        except BaseException:
            if logger:
                logger.warning("WARNING: Normalizing {} failed".format(text))
            else:
                print("WARNING: Normalizing {} failed".format(text))
            return None

        return text
