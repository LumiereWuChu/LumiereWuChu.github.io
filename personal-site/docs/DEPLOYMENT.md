# GitHub Pages 与 Sites 发布

## 免费网址：已查证

GitHub Pages 对公开仓库提供免费静态托管，并提供 `<username>.github.io` 子域名。此账号对应的网址是 **https://lumierewuchu.github.io/**。现有仓库名称已符合个人主页规则，因此不必再创建一个仓库，也不必购买域名。

如果以后想使用自己的 `.com`、`.me` 等独立域名，需要自行注册该域名，再绑定到 GitHub Pages；免费 `github.io` 地址可直接用于简历、个人名片和求职。

本项目使用静态页面与静态媒体，适合 GitHub Pages。创作台在本地编辑并导出内容，提交到仓库后自动构建；浏览者无需登录即可访问公开的 GitHub Pages 站点。Sites 当前预览仍为本人可访问，两个地址的访问范围不同。

准备公开新版时，直接打开 [这个仓库的 Pages 设置](https://github.com/LumiereWuChu/LumiereWuChu.github.io/settings/pages)，将 Source 设为 **GitHub Actions**，再合并 [迁移 PR](https://github.com/LumiereWuChu/LumiereWuChu.github.io/pull/1)。本次更新保留 PR 审阅步骤；没有自动替换旧的公共主页。

参考：[GitHub Pages 快速入门](https://docs.github.com/en/pages/quickstart)、[自定义域名说明](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/about-custom-domains-and-github-pages)。核验日期：2026-11-24。

## 已有站点

- GitHub 账号：`LumiereWuChu`
- 已有仓库：`LumiereWuChu/LumiereWuChu.github.io`
- 默认分支：`master`
- GitHub Pages 地址：https://lumierewuchu.github.io/
- 现有站点使用 Hugo。原部署工作流在 pull request 上也会写 `gh-pages`，因此迁移 PR 必须同时替换 `.github/workflows/deploy.yml`。

## 此版本的安排

完整新源码加入 `personal-site/`。保留旧 `content/`、`assets/`、`config/` 和 Hugo 配置，方便查阅与回退。修改后的部署工作流在 PR 阶段只构建、检查链接、运行导入测试，不部署公共主页。

准备好切换主页时：

1. 审阅新站与迁移 PR。
2. 在仓库 Settings → Pages → Source 选择 **GitHub Actions**。
3. 合并迁移 PR 到 `master`。
4. 在 Actions 查看 “Build and publish Lumière” 完成；成功后访问 GitHub Pages 地址。

后续更新只需提交 `personal-site/content/` 的审核后内容、媒体及必要源码。更换域名时同步 `SITE_URL`。此版本未配置自定义域名。

## 独立 Sites 版本

Sites 使用独立源仓库和 `.openai/hosting.json` 标识。新 Sites 默认仅自己可访问；当前受众不会因 GitHub PR 而改变。GitHub Pages 发布与 Sites 发布彼此独立，修改本地文件不自动同步两端。

## 验证与回退

- 迁移 PR 的构建通过不等于公共 GitHub Pages 已上线；以部署结果为准。
- 旧 Hugo 源码保留。若要回到旧版，恢复旧工作流并将 Pages Source 恢复为原 `gh-pages` 分支设置，再按旧流程构建。
- 本地创作台不含 GitHub token，没有公开管理接口。
- 不提交 `content/backups/`、`drafts/`、本地导出包、`output/` 或 `.openai/` 到个人公共仓库。

工作流依据 [GitHub 官方自定义工作流说明](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) 与 [发布源设置说明](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)。

# English

GitHub Pages provides free static hosting for public repositories and a `<username>.github.io` subdomain. This account's address is **https://lumierewuchu.github.io/**, and the existing repository already has the correct name. No separate domain purchase is required. A custom `.com` or `.me` domain must be registered separately if desired.

Open [repository Pages settings](https://github.com/LumiereWuChu/LumiereWuChu.github.io/settings/pages), select **GitHub Actions**, then merge [the migration PR](https://github.com/LumiereWuChu/LumiereWuChu.github.io/pull/1) when ready to switch the public homepage. The PR remains reviewable; the current public homepage is not replaced automatically by this update. This follows [GitHub's quickstart](https://docs.github.com/en/pages/quickstart) and [custom-domain documentation](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/about-custom-domains-and-github-pages), checked on 24 November 2026.

The existing repository is `LumiereWuChu/LumiereWuChu.github.io` on branch `master`. The new source lives in `personal-site/`; the original Hugo files remain available. The migration also replaces `.github/workflows/deploy.yml`, because the old workflow could publish during pull-request events.

The replacement workflow only builds and validates pull requests. To switch the public site, review the migration PR, set Settings → Pages → Source to **GitHub Actions**, merge into `master`, and confirm the deployment succeeds. Its public address is https://lumierewuchu.github.io/ . Subsequent commits to `personal-site/` rebuild the site automatically.

The separate Sites deployment keeps its existing audience and has a different source repository. Local edits do not automatically synchronize Sites and GitHub. The local editor only exports files; it stores no GitHub credentials and has no public administration endpoint.

Keep local backups, drafts, exported bundles, QA screenshots, and Sites identity files out of the public repository. To restore the original site, restore the Hugo workflow and the original `gh-pages` publishing source. Refer to the official GitHub documentation linked above.
