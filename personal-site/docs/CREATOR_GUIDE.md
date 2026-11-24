# 创作与上传 / Creating and uploading

## 中文

双击 `tools/editor.html` 打开本地创作台。右上角可切换完整的中文 / English 界面。工具不联网，也不会自动发布内容。

### 新增笔记、诗歌、摄影和音乐

1. 选择栏目，填写日期和唯一的页面名称，例如 `autumn-light`。
2. 填写中文与英文的标题、简介和正文。切换界面语言可预览对应版本的文字。正文可使用博客支持的 Markdown。
3. 摄影需要上传照片；音乐需要上传音频。笔记和诗歌可以选择配图。填写中英文媒体说明，便于读者和辅助阅读工具理解作品。
4. 点击“导出内容包”，保存下载的 JSON 文件。
5. 在博客项目目录打开终端，运行：

```powershell
node scripts/import-entry.mjs "C:\Users\你的用户名\Downloads\autumn-light.json"
node scripts/build.mjs
node scripts/serve.mjs 4173
```

打开 `http://127.0.0.1:4173/` 检查页面。构建成功只代表生成了本地网页；GitHub 发布按照项目 README 中的发布步骤执行。

“保存文字草稿”只写入当前浏览器的本地存储，附件不随草稿保存。恢复后请重新选择照片或音频。切换浏览器或清理浏览器数据会丢失本地草稿；导出的 JSON 内容包包含文字和附件，可用于备份与迁移。

### 更新已有内容

使用原有的页面名称导出修改后的内容包，再明确传入 `--replace`：

```powershell
node scripts/import-entry.mjs "C:\Users\你的用户名\Downloads\autumn-light.json" --replace
node scripts/build.mjs
```

导入器会先将旧内容写入 `content/backups/`，再更新条目与附件。两个不同作品不能共用同一附件文件名；创作台默认按页面名称生成附件名。

### 上传 CV

在创作台选择“更新 CV”，上传 PDF，然后导出 `cv-update.json`。运行同一个导入命令即可把 PDF 存入 `content/uploads/cv.pdf`，并更新 `content/profile.json` 的 `cvFile` 下载路径。再次更新时加 `--replace`。

```powershell
node scripts/import-entry.mjs "C:\Users\你的用户名\Downloads\cv-update.json"
node scripts/build.mjs
```

网站中英文版本共用这个 PDF。若希望发布双语 CV，可在同一份 PDF 中放入两种语言。

### 文件与格式

| 类型 | 支持格式 | 大小限制 |
| --- | --- | --- |
| 照片 / 配图 | JPG、PNG、WebP、GIF、AVIF | 每个文件 12 MB |
| 音乐 | MP3、M4A、OGG、WAV | 每个文件 25 MB |
| CV | PDF | 10 MB |

每篇内容包支持一张图片或一段音频。大相册可以分为多个摄影条目；音乐配套说明写在正文。标签在两个语言版本中保持原文，可用英文主题词或兼顾两种语言的标签。

内容存放在 `content/entries.json`，附件存放在 `content/uploads/`。日期保留原始作品或文章的日期。导入时会检查双语字段、页面名称、日期、文件格式、路径和覆盖冲突；检查失败不会改动现有内容。

创作台是本地文件编辑工具，不是在线账户或后台。源代码中的 `tools/`、`content/backups/` 和导入包不需要公开到网页资源目录。

## English

Open `tools/editor.html` in your browser. The language button switches the full editing interface between Chinese and English. This tool runs locally and never publishes automatically.

### Add a note, poem, photograph or music recording

1. Choose a collection, date and unique page slug, such as `autumn-light`.
2. Enter the title, introduction and body in both Chinese and English. Switch the interface language to preview either text. The body may use the Markdown supported by the blog.
3. Photography requires an image; music requires an audio file. Notes and poetry may include an optional image. Provide a media description in both languages for accessible reading.
4. Select **Export entry** and save the JSON bundle.
5. From the blog project folder, run:

```powershell
node scripts/import-entry.mjs "C:\Users\your-name\Downloads\autumn-light.json"
node scripts/build.mjs
node scripts/serve.mjs 4173
```

Review `http://127.0.0.1:4173/`. A successful build creates local pages. Follow the project README to publish to GitHub.

**Save text draft** uses this browser's local storage and excludes attachments. Reselect your image or audio after restoring a draft. Changing browsers or clearing browser data may remove drafts; the exported JSON bundle includes both text and media and is the portable backup.

### Update a published entry

Keep the same slug, export your revision, and explicitly request replacement:

```powershell
node scripts/import-entry.mjs "C:\Users\your-name\Downloads\autumn-light.json" --replace
node scripts/build.mjs
```

The importer first backs up the prior content in `content/backups/`. Two different entries cannot share a media filename; the editor names attachments after the entry slug.

### Update your CV

Choose **Update CV**, select a PDF, and export `cv-update.json`. Use the same import command. It writes `content/uploads/cv.pdf` and updates `cvFile` in `content/profile.json`, preserving your other profile fields. Add `--replace` when updating an existing PDF.

```powershell
node scripts/import-entry.mjs "C:\Users\your-name\Downloads\cv-update.json"
node scripts/build.mjs
```

Both website languages link to this PDF. A bilingual CV can contain both language versions in the same document.

### Supported files

| Type | Formats | Limit |
| --- | --- | --- |
| Photograph / illustration | JPG, PNG, WebP, GIF, AVIF | 12 MB per file |
| Music | MP3, M4A, OGG, WAV | 25 MB per file |
| CV | PDF | 10 MB |

Each bundle supports one image or audio attachment. Split a larger photo series into multiple entries. Describe a musical work in its body text. Tags retain their original wording in both languages.

Entries live in `content/entries.json`; attachments live in `content/uploads/`. Preserve the original publication or creation date. The importer validates translations, slugs, dates, formats, paths and overwrite conflicts before changing content.

The studio is a local authoring tool, not an online account or administration panel. Keep `tools/`, `content/backups/` and exported bundles out of the public web-assets folder.
