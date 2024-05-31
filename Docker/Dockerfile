FROM tungyenchiang/resample-image

RUN apt-get update && \
    apt-get install -y build-essential wget unzip zip git cmake yasm \
    bzip2 \
    pkg-config \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libpq-dev && \
    libopenblas-dev && \
    apt-get -y clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install gconf-service libasound2 libatk1.0-0 libc6 libcairo2 libcups2 libdbus-1-3 libexpat1 \ 
    libfontconfig1 libgcc1 libgconf-2-4 libgdk-pixbuf2.0-0 libglib2.0-0 libgtk-3-0 \
    libnspr4 libpango-1.0-0 libpangocairo-1.0-0 libstdc++6 libx11-6 libx11-xcb1 \
    libxcb1 libxcomposite1 libxcursor1 \
    libxdamage1 libxext6 libxfixes3 libxi6 \
    libxrandr2 libxrender1 libxss1 libxtst6 ca-certificates fonts-liberation \
    libappindicator1 libnss3 lsb-release xdg-utils wget -y && \
    apt-get -y clean && \
    rm -rf /var/lib/apt/lists/*

# RUN FORCE_CUDA=1 pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

WORKDIR /save
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/CityU-AIM-Group/EPT.git && \
    cd EPT && \
    cd point_transformer_lib && \
    python setup.py build_ext install




