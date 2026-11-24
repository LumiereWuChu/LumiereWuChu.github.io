# GitHub Pages 与 Sites 发布

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

The existing repository is `LumiereWuChu/LumiereWuChu.github.io` on branch `master`. The new source lives in `personal-site/`; the original Hugo files remain available. The migration also replaces `.github/workflows/deploy.yml`, because the old workflow could publish during pull-request events.

The replacement workflow only builds and validates pull requests. To switch the public site, review the migration PR, set Settings → Pages → Source to **GitHub Actions**, merge into `master`, and confirm the deployment succeeds. Its public address is https://lumierewuchu.github.io/ . Subsequent commits to `personal-site/` rebuild the site automatically.

The separate Sites deployment keeps its existing audience and has a different source repository. Local edits do not automatically synchronize Sites and GitHub. The local editor only exports files; it stores no GitHub credentials and has no public administration endpoint.

Keep local backups, drafts, exported bundles, QA screenshots, and Sites identity files out of the public repository. To restore the original site, restore the Hugo workflow and the original `gh-pages` publishing source. Refer to the official GitHub documentation linked above.
