"""Check real Sarvam tokenization on every input without downloading model weights."""
from pathlib import Path
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    os.environ['HASTIKA_DATA_DIR'] = str(args.data_dir.resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from transformers import AutoConfig, AutoTokenizer
    from llm_config import LLM_TRAINING, LLM_TRAINING_B, TASK_A, TASK_B
    from src.data import load_epoch_validation_frame, load_task_frames, prompt_inputs, resolve_data_file
    from src.llm_data import ChatEncoder

    config = AutoConfig.from_pretrained(
        LLM_TRAINING.model_name, revision=LLM_TRAINING.model_revision,
        trust_remote_code=False,
    )
    if config.model_type != 'llama':
        raise ValueError('This preflight expects the Sarvam-1 Llama backbone.')
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_TRAINING.model_name, revision=LLM_TRAINING.model_revision,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    result = {'model': LLM_TRAINING.model_name, 'revision': LLM_TRAINING.model_revision,
              'architecture': config.architectures, 'tasks': {}}
    for name, task, training_config in [('a', TASK_A, LLM_TRAINING), ('b', TASK_B, LLM_TRAINING_B)]:
        train, prediction = load_task_frames(task)
        assert not (set(train.id) & set(prediction.id)), 'Train/open-validation IDs overlap.'
        encoder = ChatEncoder(tokenizer, task, training_config.max_length)
        checked = {}
        for split, frame in [('train', train), ('open_validation', prediction)]:
            truncated = maximum = 0
            for pair in prompt_inputs(frame, task):
                prompt, cut = encoder.encode_prompt(pair)
                truncated += int(cut)
                maximum = max(maximum, len(prompt) + encoder.response_reserve)
                decoded = tokenizer.decode(prompt, skip_special_tokens=False)
                if task.include_english_translation:
                    assert decoded.index('English translation:') < decoded.index(
                        'Comment (raw code-mixed):' if task.prompt_style == 'khangtest' else 'Original code-mixed comment:')
                for label in range(len(task.labels)):
                    feature = encoder.feature(prompt, label)
                    assert len(feature['input_ids']) <= training_config.max_length
                    assert feature['labels'][:len(prompt)] == [-100] * len(prompt)
                    assert feature['input_ids'][len(prompt):] == encoder.completions[label]
                    assert tokenizer.eos_token_id in encoder.completions[label]
            checked[split] = {'rows': len(frame), 'truncated': truncated, 'max_full_chat_tokens': maximum}
        if name == 'b':
            gold = load_epoch_validation_frame(task)
            assert gold.id.tolist() == prediction.id.tolist()
        result['tasks'][name] = {
            'data': checked,
            'open_validation_file': str(resolve_data_file(task.prediction_filename)),
            'selection_train_rows': int(train.fold.ne(5).sum()),
            'holdout_rows': int(train.fold.eq(5).sum()),
            'completion_tokens': dict(zip(task.labels, [len(c) for c in encoder.completions])),
            'completion_text': dict(zip(task.labels, [tokenizer.decode(c) for c in encoder.completions])),
            'training_excluded_rows': train.attrs.get('training_exclusions', {}).get('excluded_rows', 0),
        }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
