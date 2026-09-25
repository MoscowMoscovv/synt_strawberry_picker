# Blender 4.3.2's official Linux archive is x86-64 (use linux/amd64).
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BLENDER_EXE=/opt/blender/blender

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash ca-certificates curl xz-utils \
        libegl1 libgl1 libx11-6 libxext6 libxfixes3 libxi6 libxkbcommon0 \
        libxrender1 libxxf86vm1 libsm6 libice6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# SHA-256 from https://download.blender.org/release/Blender4.3/blender-4.3.2.sha256
RUN curl --fail --location --retry 3 \
        https://download.blender.org/release/Blender4.3/blender-4.3.2-linux-x64.tar.xz \
        --output /tmp/blender.tar.xz \
    && echo "4da1c956673c0485e63054e563ee69198cc8f80d8157dd7592dffc8a6a5592e6  /tmp/blender.tar.xz" | sha256sum --check - \
    && mkdir -p /opt/blender \
    && tar -xJf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 \
    && rm /tmp/blender.tar.xz \
    && ln -s /opt/blender/blender /usr/local/bin/blender \
    && blender --background --factory-startup --python-exit-code 1 \
        --python-expr "import bpy; assert bpy.app.version == (4, 3, 2), bpy.app.version"

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt \
    && python -m pip check

COPY plant_generator/ ./plant_generator/
COPY tests/ ./tests/
COPY batch_generate_strawberries.py generate_plant_grid.py make_crown.py \
    view_strawberry_mujoco.py strawberry_variants.example.json LICENSE ./
RUN mkdir -p /app/dataset /app/Strawberry /output

CMD ["bash"]
