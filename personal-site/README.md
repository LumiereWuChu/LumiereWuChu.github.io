# Lumière — 吴楚 / Chu Wu

完整中英文个人博客：光子计算研究、论文、阅读手记、音乐、诗歌、摄影与研究简历。零安装依赖，Node.js 20+ 即可构建。中文与英文页面各有独立地址，语言按钮保留当前内容页。

## 本地运行

```text
node scripts/build.mjs
node scripts/check.mjs
node scripts/serve.mjs 4173
```

浏览器打开 http://127.0.0.1:4173/ 。终端 Ctrl+C 停止预览。

## 添加自己的作品

双击 `tools/editor.html` 打开本地创作台。可编辑中英文笔记或诗歌、选择摄影或音频、导出内容包；也可导入一份自己的 PDF CV。创作台不上传数据，不是公开网站的后台。

```text
node scripts/import-entry.mjs "完整路径/作品内容包.json"
node scripts/build.mjs
node scripts/check.mjs
```

详细说明见 [创作指南](docs/CREATOR_GUIDE.md)。媒体上限：图片 12 MB、音频 25 MB、CV 10 MB。真正发布需把审核后的源码及媒体提交到 GitHub；本地草稿和导出操作不会自动公开。

## 内容文件

| 文件 | 修改内容 |
| --- | --- |
| `content/profile.json` | 姓名、简介、公开联系信息、CV 文件路径 |
| `content/projects.json` | 研究项目，双语介绍 |
| `content/publications.json` | 论文、作者、原始来源和发表状态 |
| `content/entries.json` | 笔记、诗歌、摄影、音乐，双语正文 |
| `content/uploads/` | 自己上传的图像、音频和 CV |
| `src/style.css` | 布局、字体、颜色、打印样式 |

Markdown 正文支持段落、标题、无序列表、引用、加粗、行内代码、代码块和 HTTPS 链接。当前不使用第三方数学渲染器，精确公式可使用代码块。原始 HTML 不执行。

## GitHub 发布

新站源码放进现有仓库的 `personal-site/`，通过单独迁移分支和 PR 审阅。原 Hugo 内容保留在原路径。将本目录 `deployment/github-pages.yml` 作为仓库根目录 `.github/workflows/deploy.yml` 的替代版本。新工作流在 PR 上只构建与校验，只有 `master` 合并后的推送或在 `master` 手动运行才部署。

仓库 Settings → Pages → Build and deployment → Source 需选择 **GitHub Actions**。合并迁移 PR 后，目标地址是 https://lumierewuchu.github.io/ 。配置和具体迁移步骤见 [发布说明](docs/DEPLOYMENT.md)。

网站构建结果在 `dist/`。Sites 独立预览通过 `.openai/hosting.json` 配置；向 GitHub 提交时不要复制 Sites 身份文件。`SITE_URL` 环境变量决定 canonical、站点地图与分享链接。GitHub 工作流已设为正确地址。

## 已有内容与边界

- 公开身份、4 条论文记录根据实验室、出版商和 arXiv 核验；预印本、期刊研究论文和评论分别标记。
- 旧站一篇已发布的读书笔记迁为双语整理版，保留日期与原文链接。
- GenMPU 和光辅助 KV-cache 使用研究方向简介，不公开内部性能数据。
- 音乐、诗歌、摄影尚无本人上传作品，页面如实显示待收录状态。
- CV 页面是已核实资料的可打印研究概要，不假造教育年份、获奖或任职经历。可通过创作台替换为个人完整 PDF。
- 首页图像为本项目生成的 SVG 光影视觉，不是实验场图或个人摄影。

## 验证

```text
node scripts/build.mjs
node scripts/check.mjs
node --test tests/import-entry.test.mjs
```

浏览器检查及截图记录在本地 `output/playwright/`，不进入发布包。

## English

This dependency-free bilingual site contains research directions, verified publications, a migrated reading note, creative collections, and a printable research CV. Run `node scripts/build.mjs`, `node scripts/check.mjs`, then `node scripts/serve.mjs 4173`.

Open `tools/editor.html` locally to create bilingual notes, poems, photo or music entries, or upload your PDF CV. Export a JSON bundle, run `node scripts/import-entry.mjs path/to/bundle.json`, rebuild, review, then commit the reviewed source and media. Local draft saving does not publish anything. See the [bilingual creator guide](docs/CREATOR_GUIDE.md).

For the existing GitHub repository, keep this project under `personal-site/` and replace the root `.github/workflows/deploy.yml` with `deployment/github-pages.yml`. Pull requests run checks only. Publishing occurs on `master` after merge; set the repository Pages source to **GitHub Actions**. The original Hugo source remains intact. See [deployment notes](docs/DEPLOYMENT.md).

Original writing and uploaded works remain the author's property. No sample third-party photographs or invented creative works are presented as the owner's work.
