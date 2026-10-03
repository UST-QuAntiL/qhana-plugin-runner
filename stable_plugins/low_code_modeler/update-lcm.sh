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

if ! git diff --cached --quiet; then
  echo "you have local staged changes, please stash them before running this script" 1>&2
  exit 1
fi

git rm -rf static
git reset -- static/workarounds.js
git checkout -- static/workarounds.js
pushd "$low_code_modeler"
commit="$(git rev-parse --verify HEAD)"
pnpm install
pnpm run build --outDir "$plugin_folder"/static
popd
sed -i -e '/<\/script>/r head.html' -e '/<\/div>/r body.html' static/index.html
printf "%s" "$commit" > static/.git-revision
git add static

git commit -m "low-code-modeler: update static files to commit $commit (using update-lcm.sh)"
