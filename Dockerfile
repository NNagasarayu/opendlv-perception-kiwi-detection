# Copyright (C) 2018  Christian Berger
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

FROM ubuntu:20.04 as builder
ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update --fix-missing
RUN apt-get install -y \
    build-essential \
    cmake \
    software-properties-common \
    libopencv-dev

RUN add-apt-repository -y ppa:chrberger/libcluon
RUN apt-get update
RUN apt-get install -y libcluon

ADD . /opt/sources
WORKDIR /opt/sources
RUN mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/tmp/dest .. && \
    make && make install


FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update
RUN apt-get install -y \
    libopencv-core4.2 \
    libopencv-imgproc4.2 \
    libopencv-video4.2 \
    libopencv-calib3d4.2 \
    libopencv-features2d4.2 \
    libopencv-objdetect4.2 \
    libopencv-highgui4.2 \
    libopencv-videoio4.2 \
    libopencv-flann4.2 \
    libopencv-dnn4.2 \
    python3-opencv

WORKDIR /usr/bin
COPY --from=builder /tmp/dest /usr
COPY --from=builder /opt/sources/data /usr/data
ENTRYPOINT ["/usr/bin/opendlv-perception-kiwi-detection"]

