#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

sed -i '$a\\n' ../../demo/docs/zh_cn/*_demo.md
cat ../../demo/docs/zh_cn/*_demo.md | sed "s/^## 2D\(.*\)Demo/##\1Estimation/" | sed "s/md###t/html#t/g" | sed '1i\# Demos\n' | sed 's=](/docs/en/=](/=g' | sed 's=](/=](https://github.com/open-mmlab/mmpose/tree/dev-1.x/=g' >demos.md

 # remove /docs_zh-CN/ for link used in doc site
sed -i 's=](/docs/zh_cn/=](=g' ./tutorials/*.md
sed -i 's=](/docs/zh_cn/=](=g' ./tasks/*.md
sed -i 's=](/docs/zh_cn/=](=g' ./papers/*.md
sed -i 's=](/docs/zh_cn/=](=g' ./topics/*.md
sed -i 's=](/docs/zh_cn/=](=g' data_preparation.md
sed -i 's=](/docs/zh_cn/=](=g' get_started.md
sed -i 's=](/docs/zh_cn/=](=g' install.md
sed -i 's=](/docs/zh_cn/=](=g' benchmark.md
# sed -i 's=](/docs/zh_cn/=](=g' changelog.md
sed -i 's=](/docs/zh_cn/=](=g' faq.md

sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' ./tutorials/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' ./tasks/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' ./papers/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' ./topics/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' data_preparation.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' get_started.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' install.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' benchmark.md
# sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' changelog.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' faq.md
