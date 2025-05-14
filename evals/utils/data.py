from pathlib import Path
from typing import Any, Iterable, Optional, Union


def load_jsonl(input_file: Union[str, Path], x_type: Optional[Any] = None):
    import msgspec

    input_file = Path(input_file)
    assert input_file.exists(), f"File {input_file} does not exist"
    assert input_file.name.endswith(".jsonl")
    if x_type is not None:
        x_decoder = msgspec.json.Decoder(x_type)
    else:
        x_decoder = msgspec.json.Decoder()
    with open(input_file) as fh:
        for line in fh:
            try:
                yield x_decoder.decode(line.encode("utf-8"))
            except Exception as e:
                print(f"Error while decoding line in {str(input_file)}: {line}")
                raise e


def save_jsonl(
    xs: Iterable[Any],
    output_file: Union[str, Path],
    x_type: Optional[Any] = None,
    exists_ok=False,
):
    import msgspec

    output_file = Path(output_file)
    assert output_file.name.endswith(".jsonl")
    if not exists_ok:
        assert not output_file.exists(), f"File {output_file} already exists"
    x_encoder = msgspec.json.Encoder()
    with open(output_file, "wb") as fh:
        n_lines = 0
        for x in xs:
            if x_type is not None:
                assert isinstance(x, x_type)
            fh.write(x_encoder.encode(x))
            fh.write(b"\n")
            n_lines += 1
    if n_lines == 0:
        print(f"Warning: no lines written to {output_file}")
        # Remove
        output_file.unlink()
