FROM python:3.7

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /genn

# Add setup.py to install dependecies and cache them
COPY setup.py /genn
COPY README.md /genn
COPY genn/bin/genn /genn/genn/bin/genn
COPY genn/version.py /genn/genn/version.py
RUN pip install -e /genn

# Add source code
# Everything after this line is uncached
COPY . /genn

RUN pip install -e /genn

ENTRYPOINT ["genn"]

CMD ["--help"]