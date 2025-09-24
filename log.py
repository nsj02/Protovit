"""Simple logging utility returning a print/write callable."""

import os


def create_logger(log_filename: str, display: bool = True):
    """파일 로거 생성.

    - 입력: 로그 파일 경로, stdout 출력 여부
    - 출력: `(logger_fn, close_fn)`
        * `logger_fn(text)` 호출 시 파일에 append, `display=True`면 stdout에도 출력
        * 내부적으로 10줄마다 `flush` + `os.fsync` 수행
        * `close_fn()`으로 파일을 닫을 수 있음
    """

    f = open(log_filename, 'a')
    counter = [0]

    def logger(text: str):
        if display:
            print(text)
        f.write(text + '\n')
        counter[0] += 1
        if counter[0] % 10 == 0:
            f.flush()
            os.fsync(f.fileno())

    return logger, f.close
