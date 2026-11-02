# brancusi-engine tasks. Run `just` to see the list.

default:
    @just --list --unsorted

# Bump the version (patch: 0.1.1 -> 0.1.2), commit, push, tag and watch the release CI
bump part='patch':
    #!/usr/bin/env bash
    set -euo pipefail
    branch=$(git branch --show-current)
    if [ "$branch" != "brancusi" ]; then echo "Use the brancusi branch (now: $branch)."; exit 1; fi
    if [ -n "$(git status --porcelain)" ]; then echo "Commit or stash your changes first."; exit 1; fi
    git pull --ff-only
    # The fork, not upstream: gh picks the parent repo by default.
    repo=$(git remote get-url origin | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##')
    file=brancusi/CMakeLists.txt
    old=$(sed -nE 's/^set\(BRANCUSI_VERSION ([0-9]+\.[0-9]+\.[0-9]+)\)/\1/p' "$file")
    IFS=. read -r major minor patch <<< "$old"
    case "{{part}}" in
      patch) patch=$((patch + 1)) ;;
      minor) minor=$((minor + 1)); patch=0 ;;
      major) major=$((major + 1)); minor=0; patch=0 ;;
      *) echo "part must be patch, minor or major."; exit 1 ;;
    esac
    new="$major.$minor.$patch"
    tag="brancusi-v$new"
    if git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then echo "The tag $tag exists already."; exit 1; fi
    sed -i.bak -E "s/^set\(BRANCUSI_VERSION $old\)/set(BRANCUSI_VERSION $new)/" "$file" && rm "$file.bak"
    printf '\n---\n%s\n\n**Version %s.** Release build.\n' "$(date '+%H:%M %A %-d %B %Y')" "$new" >> CHANGELOG.md
    git add "$file" CHANGELOG.md
    git commit -q -m "Bump the version to $new"
    git push -q origin brancusi
    git tag "$tag"
    git push -q origin "$tag"
    echo "Pushed $tag. Waiting for CI..."
    sleep 10
    run=$(gh run list --repo "$repo" --workflow brancusi-engine.yml --branch "$tag" --limit 1 --json databaseId -q '.[0].databaseId')
    gh run watch "$run" --repo "$repo" --exit-status
    gh release view "$tag" --repo "$repo" --json url -q .url
