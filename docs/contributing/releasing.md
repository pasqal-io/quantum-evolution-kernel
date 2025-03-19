# Releasing a new version

You only need to read this document if you're a maintainer of this library.

Steps to release a new version:

1. Bump up the version number in pyproject.toml (e.g. `0.7.9`).
2. Bump up the links in the `README.md` that point to specific versions of the documentation.
3. Open a Pull Request.
4. Get it merged.
5. Optionally, test a pre-release
    1. Open the GitHub "draft new release" panel.
    2. Create a new tag (e.g. `v0.7.9rc1`).
    3. Mark the release as "pre-release".
    4. Confirm the release.
    5. Check in the Actions that the `publish` action has succeeded.
6. Actually release.
    1. Open the GitHub "draft new release" panel.
    2. Create a new tag (e.g. `v0.7.9rc1`).
    3. Mark the release as "latest release".
    4. Confirm the release.
    5. Check in the Actions that the `publish` action has succeeded.
