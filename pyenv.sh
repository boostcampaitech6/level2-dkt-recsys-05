### python 컴파일에 필요한 패키지들을 설치
PACKAGES='curl libbz2-dev libssl-dev libsqlite3-dev liblzma-dev libffi-dev libncursesw5-dev libreadline-dev build-essential wget locales libgdbm-dev libnss3-dev zlib1g-dev'
apt update
apt install $PACKAGES -y
curl https://pyenv.run | bash

### PATH에 pyenv 등록
echo 'export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"' >> ~/.bashrc

source ~/.bashrc

### locale 설정
locale-gen en_US.UTF-8
update-locale LANG=en_US.UTF-8

### python 3.10.13 설치
pyenv install 3.10.13
pyenv global 3.10.13

### python 버전 확인
which python