#!/usr/bin/env bash

set -euo pipefail
set -x

if ! test -f update-lcm.sh; then
  echo "you must execute $0 while you are in the folder containing $0" 1>&2
  exit 1
fi

plugin_folder="$PWD"

if ! test "$#" -eq 1; then
  echo "USAGE: $0 <path to low-code-modeler repo>" 1>&2
  exit 1
fi

low_code_modeler="$1"

if ! test -d "$low_code_modeler"; then
  echo "USAGE: $0 <path to low-code-modeler repo>" 1>&2
  exit 1
fi

if ! test -d "$low_code_modeler/.git"; then
  echo "low-code-modeler repo must be a git repository"
fi

type git
type pnpm

stashed=""
git diff --cached --quiet || stashed=1
if test -n "$stashed"; then
  echo "stashing local changes" 1>&2
  echo "if this script fails, you have to undo this with git stash pop" 1>&2
  git stash
fi

git rm -rf static
git reset -- static/workarounds.js
git checkout -- static/workarounds.js
pushd "$low_code_modeler"
commit="$(git rev-parse --verify HEAD)"
pnpm install
pnpm run build --outDir "$plugin_folder"/static
popd
sed -i '/<\/script>/a <script defer src="/static/microfrontend.js"></script><script src="workarounds.js"></script>' static/index.html
printf "%s" "$commit" > static/.git-revision
git add static

git commit -m "low-code-modeler: update static files to commit $commit (using update-lcm.sh)"

if test -n "$stashed"; then
  git stash pop
fi
