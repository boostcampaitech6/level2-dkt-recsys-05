# Environment Setting Guide

할당받은 서버에서 pyenv.sh을 실행하면 pyenv가 설치됩니다.

```
$ bash pyenv.sh
$ source ~/.bashrc
```

poetry를 설치하고 cache 디렉토리를 변경해주세요.
```
$ poetry config cache-dir /data/ephemeral/.cache/pypoetry
```