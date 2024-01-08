### python 컴파일에 필요한 패키지들을 설치
apt install curl libbz2-dev libssl-dev libsqlite3-dev liblzma-dev libffi-dev libncursesw5-dev libreadline-dev

### pyenv 설치
curl https://pyenv.run | bash

### python 3.10.13 설치
pyenv install 3.10.13
pyenv global 3.10.13

### python 버전 확인
which python