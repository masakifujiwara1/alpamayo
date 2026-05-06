## Alpamayo 1 OSS コントリビューションルール

#### Issue 管理

- 機能追加、バグ修正、変更リクエストはすべて、まず [Alpamayo 1 Issue Request](https://github.com/NVlabs/alpamayo/issues) を作成するところから始めてください。
  - Issue request は、コードレビュー前に Alpamayo 1 の研究者によるレビューと承認が必要です。

#### コーディングガイドライン

- 新しいコードを追加する場合や既存機能を拡張・修正する場合は、該当するファイル、サブモジュール、モジュール、プロジェクトの既存規約に従ってください。

- コードフォーマットとスタイルの一貫性を保つため、提供されている設定ファイルを使って、変更したソースに対して `pre-commit format` も実行してください。これにより、Alpamayo 1 のコードフォーマット規則が次の項目に適用されます。

  - クラス、関数/メソッド、変数/フィールドの命名
  - コメントスタイル
  - インデント
  - 行長

- 保守性と可読性を保つため、既存コードに不要な複雑さを持ち込まないでください。

- Pull Request (PR) はできるだけ簡潔に保ってください。

  - コメントアウトされたコードをコミットしないでください。
  - 可能な限り、各 PR は 1 つの関心事だけを扱うようにしてください。目的を達成するために互いに無関係な複数の修正が必要な場合は、複数の PR を作成し、説明文で依存関係を示すことを推奨します。1 つの PR に含まれる変更が複雑になるほど、レビューに時間がかかります。

- コミットタイトルは命令形で書き、[これらのルール](https://chris.beams.io/posts/git-commit/)に従い、PR に対応する Issue 番号を参照してください。コミットメッセージの推奨形式は次のとおりです。

```
#<Issue Number> - <Commit Title>

<Commit Body>
```

- ビルドログに警告やエラーがなく、クリーンであることを確認してください。

- コードを提出する前に、すべてのテストが通ることを確認してください。

- すべての OSS コンポーネントには、機能、依存関係、既知の問題を説明するドキュメント (README) を添付する必要があります。

  - 既存のサンプルやプラグインについては `README.md` を参照してください。

- すべての OSS コンポーネントには、対応するテストが必要です。

  - 新しいコンポーネントを導入する場合は、機能を検証するテストサンプルを提供してください。

- 自分の作業をオープンソースへ提供できることを確認してください (コードによってライセンスまたは特許の衝突が生じないこと)。コミットには [`sign`](#signing-your-work) が必要です。

- コントリビューションのレビューには時間がかかる場合があります。ご理解に感謝します。皆さまの貢献を歓迎しています。

#### Pull Request

コードコントリビューションの開発者ワークフローは次のとおりです。

1. 開発者はまず [upstream](https://github.com/NVlabs/alpamayo) の Alpamayo 1 OSS リポジトリを [fork](https://help.github.com/en/articles/fork-a-repo) する必要があります。

2. fork したリポジトリを git clone し、変更を個人の fork に push します。

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_FORK.git Alpamayo-1
# 対象ブランチをチェックアウトして変更をコミットする
# fork 上のブランチ (remote) にコミットを push する
git push -u origin <local-branch>:<remote-branch>
```

3. コード変更が fork 上で stage され、レビューの準備ができたら、fork のブランチから upstream の選択したブランチへ変更を merge する [Pull Request](https://help.github.com/en/articles/about-pull-requests) (PR) を [作成](https://help.github.com/en/articles/creating-a-pull-request) できます。

- PR の source branch と target branch を選ぶ際は注意してください。
- PR を作成するとコードレビューのプロセスが始まります。
- 少なくとも 1 名の Alpamayo 1 研究者がレビュー担当として割り当てられます。
- レビュー中の PR は、PR タイトルの先頭に `[WIP]` を付けて work-in-progress として示してください。

4. 現時点では CI/CD プロセスがないため、PR は、開発者および/またはレビュー担当の Alpamayo 1 研究者による十分な手動テストが完了した後にのみ承認され、対応する Issue が close されます。

#### Signing Your Work

- すべてのコントリビューターに、コミットへの "sign-off" を求めています。これは、その貢献があなた自身の作業であること、または同じライセンスもしくは互換ライセンスのもとで提出する権利があることを証明するものです。

  - Signed-Off されていないコミットを含むコントリビューションは受け付けられません。

- コミットに sign off するには、コミット時に `--signoff` (または `-s`) オプションを使います。

  ```bash
  $ git commit -s -m "Add cool feature."
  ```

  これにより、コミットメッセージに次の行が追加されます。

  ```
  Signed-off-by: Your Name <your@email.com>
  ```

- DCO の全文:

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```
