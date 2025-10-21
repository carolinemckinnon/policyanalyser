# preprocessing.py

import re
from collections import Counter
import math
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

DEFAULT_STOPWORDS = set(token.lower() for token in ENGLISH_STOP_WORDS)


def basic_sentence_tokenize(text):
    if not text:
        return []
    normalized = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    if not sentences:
        return [normalized]
    cleaned = [s.strip() for s in sentences if s.strip()]
    return cleaned or [normalized]

def clean_text(
    text,
    lowercase=True,
    remove_punctuation=True,
    remove_stopwords=False,
    strip_nonascii=False,
    strip_scene_elements=False,
    strip_headings=None,
    marker_tags=None,
    stopwords_list=None,
    **_ignored
):

    if marker_tags:
        for tag, opts in marker_tags.items():
            if opts.get("strip", False):
                text = text.replace(tag, "")

    if strip_headings:
        for heading in strip_headings:
            text = text.replace(heading, "")

    if lowercase:
        text = text.lower()

    if strip_nonascii:
        text = text.encode("ascii", "ignore").decode()

    if strip_scene_elements:
        text = re.sub(r"^\s*[A-Z][A-Z\s]+:\s*", "", text, flags=re.MULTILINE)  # Speaker labels
        text = re.sub(r"\[.*?\]", "", text)  # Stage directions

    if remove_punctuation:
        text = re.sub(r"[.,!?;:\"'()\[\]{}<>\-]+", " ", text)

    if remove_stopwords:
        if stopwords_list is None or stopwords_list == "english":
            stop_words = DEFAULT_STOPWORDS
        else:
            stop_words = set(word.lower() for word in stopwords_list)
        text = " ".join([word for word in text.split() if word not in stop_words])

    return text.strip()

def segment_text_with_units(
    text,
    mode="Sentence-based",
    segment_size=3,
    content_word_target=50,
    padding=True,
    stride=1,
    preprocessing_opts=None,
    strip_headings=None
):
    # Step 1: Tokenize the text into units (sentences or paragraphs)
    if mode == "Paragraph-based":
        units = [p.strip() for p in text.split("\n") if p.strip()]
    else:
        units = basic_sentence_tokenize(text)

    # Step 2: Detect sentence-level overlay markers
    overlay_tags = preprocessing_opts.get("marker_tags", {})
    sentence_level_overlays = []
    for i, sentence in enumerate(units):
        for tag, opts in overlay_tags.items():
            if opts.get("overlay") and tag in sentence:
                sentence_level_overlays.append((i, tag))

    # Step 3: Segment the text
    if mode == "Evenly-sized content-word groups":
        segs, cleaned, ranges = segment_by_target_term_count(
            units,
            content_word_target,
            preprocessing_opts,
            padding=padding,
            stride=stride,
            strip_headings=strip_headings
        )
    else:
        original_segments = []
        cleaned_segments = []
        ranges = []

        step = stride if padding else segment_size
        limit = len(units) - segment_size + 1 if padding else len(units)

        for i in range(0, limit, step):
            chunk = units[i:i+segment_size]
            if not chunk:
                continue
            original = " ".join(chunk)
            cleaned = clean_text(original, **preprocessing_opts, strip_headings=strip_headings)
            original_segments.append(original)
            cleaned_segments.append(cleaned)
            ranges.append((i, i + len(chunk) - 1))

        segs = original_segments
        cleaned = cleaned_segments

    # Step 4: Map each sentence index to its segment index
    overlay_indices = []
    for sent_idx, label in sentence_level_overlays:
        for seg_idx, (start, end) in enumerate(ranges):
            if start <= sent_idx <= end:
                overlay_indices.append((seg_idx, label))
                break

    # Step 5: Return the results
    return segs, cleaned, ranges, overlay_indices

def segment_by_target_term_count(sentences, target_terms, preprocessing_opts, padding=False, stride=1, strip_headings=None):
    original_segments = []
    cleaned_segments = []
    ranges = []

    if not padding:
        current_original = []
        current_count = 0
        start_idx = 0

        for idx, sentence in enumerate(sentences):
            cleaned = clean_text(sentence, **preprocessing_opts, strip_headings=strip_headings)
            term_count = len(cleaned.split())

            if current_count == 0:
                start_idx = idx

            current_original.append(sentence)
            current_count += term_count

            if current_count >= target_terms:
                original = " ".join(current_original)
                cleaned_seg = clean_text(original, **preprocessing_opts, strip_headings=strip_headings)
                original_segments.append(original)
                cleaned_segments.append(cleaned_seg)
                ranges.append((start_idx, idx))
                current_original = []
                current_count = 0

        if current_original:
            if original_segments:
                original_segments[-1] += " " + " ".join(current_original)
                cleaned_segments[-1] = clean_text(original_segments[-1], **preprocessing_opts, strip_headings=strip_headings)
                prev_start, _ = ranges[-1]
                ranges[-1] = (prev_start, len(sentences) - 1)
            else:
                original = " ".join(current_original)
                cleaned = clean_text(original, **preprocessing_opts, strip_headings=strip_headings)
                original_segments.append(original)
                cleaned_segments.append(cleaned)
                ranges.append((start_idx, len(sentences) - 1))

    else:
        for i in range(0, len(sentences) - stride + 1, stride):
            current_original = []
            current_count = 0
            start_idx = i
            for j in range(i, len(sentences)):
                cleaned = clean_text(sentences[j], **preprocessing_opts, strip_headings=strip_headings)
                term_count = len(cleaned.split())
                current_original.append(sentences[j])
                current_count += term_count
                if current_count >= target_terms:
                    break
            if current_original:
                original = " ".join(current_original)
                cleaned_seg = clean_text(original, **preprocessing_opts, strip_headings=strip_headings)
                original_segments.append(original)
                cleaned_segments.append(cleaned_seg)
                ranges.append((start_idx, start_idx + len(current_original) - 1))

    return original_segments, cleaned_segments, ranges

def remove_high_frequency_terms(segments, cutoff_percentage):
    if not segments or cutoff_percentage <= 0:
        return segments, []

    num_segments = len(segments)
    if num_segments == 0:
        return segments, []

    doc_freq = Counter()
    tokenised_segments = []
    for segment in segments:

        tokens = segment.split()
        tokenised_segments.append(tokens)
        doc_freq.update(set(tokens))

    threshold = max(1, math.ceil(num_segments * (cutoff_percentage / 100.0)))
    high_freq_terms = {word for word, count in doc_freq.items() if count >= threshold}

    if not high_freq_terms:
        return segments, []

    filtered_segments = []
    for tokens in tokenised_segments:
        filtered_tokens = [token for token in tokens if token not in high_freq_terms]
        if filtered_tokens:
            filtered_segments.append(" ".join(filtered_tokens))
        else:
            filtered_segments.append(" ".join(tokens))

    return filtered_segments, sorted(high_freq_terms)

def extract_marker_candidates(paragraphs, use_headings=True, use_heuristics=True):
    candidates = {}

    for p in paragraphs:
        if hasattr(p, "text"):
            text = p.text.strip()
            style_name = p.style.name if hasattr(p, "style") and p.style else ""
        else:
            text = str(p).strip()
            style_name = ""

        if not text:
            continue

        heading_level = None
        if use_headings and style_name.startswith("Heading"):
            heading_level = style_name
            label = f"{text} ({style_name})"
            candidates[text] = (label, heading_level)
            continue

        if use_heuristics:
            if text.isupper() or (text.istitle() and len(text.split()) <= 5):
                label = text
                candidates[text] = (label, None)

    return [(label, raw, level) for raw, (label, level) in candidates.items()]
