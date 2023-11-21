# Contribution guide

**Want to contribute? Great!**
We try to make it easy, and all contributions, even the smaller ones, are more than welcome.
This includes bug reports, fixes, documentation, examples...
But first, read this page (including the small print at the end).

## Legal

All original contributions to TrustyAI-explainability are licensed under the
[ASL - Apache License](https://www.apache.org/licenses/LICENSE-2.0),
version 2.0 or later, or, if another license is specified as governing the file or directory being
modified, such other license.

## Issues

Python TrustyAI uses [GitHub to manage and report issues](https://github.com/trustyai-explainability/trustyai-explainability-python/issues).

If you believe you found a bug, please indicate a way to reproduce it, what you are seeing and what you would expect to see. 
Don't forget to indicate your Python TrustyAI, Java, and Maven version.

### Checking an issue is fixed in main

Sometimes a bug has been fixed in the `main` branch of Python TrustyAI and you want to confirm it is fixed for your own application. 
Testing the `main` branch is easy and you can build Python TrustyAI all by yourself.

## Creating a Pull Request (PR)

To contribute, use GitHub Pull Requests, from your **own** fork.

- PRs should be always related to an open GitHub issue. If there is none, you should create one.
- Try to fix only one issue per PR.
- Make sure to create a new branch. Usually branches are named after the GitHub ticket they are addressing. E.g. for ticket "issue-XYZ An example issue" your branch should be at least prefixed with `FAI-XYZ`. E.g.:

        git checkout -b issue-XYZ
        # or
        git checkout -b issue-XYZ-my-fix

- When you submit your PR, make sure to include the ticket ID, and its title; e.g., "issue-XYZ An example issue".
- The description of your PR should describe the code you wrote. The issue that is solved should be at least described properly in the corresponding GitHub ticket.
- If your contribution spans across multiple repositories, use the same branch name (e.g. `issue-XYZ`).
- If your contribution spans across multiple repositories, make sure to list all the related PRs.

### Python Coding Guidelines

PRs will be checked against `black` and `pylint` before passing the CI.

You can perform these checks locally to guarantee the PR passes these checks.

### Requirements for Dependencies

Any dependency used in the project must fulfill these hard requirements:

- The dependency must have **an Apache 2.0 compatible license**.
    - Good: BSD, MIT, Apache 2.0
    - Avoid: EPL, LGPL
        - Especially LGPL is a last resort and should be abstracted away or contained behind an SPI.
        - Test scope dependencies pose no problem if they are EPL or LPGL.
    - Forbidden: no license, GPL, AGPL, proprietary license, field of use restrictions ("this software shall be used for good, not evil"), ...
        - Even test scope dependencies cannot use these licenses.
    - To check the ALS compatibility license please visit these links:[Similarity in terms to the Apache License 2.0](http://www.apache.org/legal/resolved.html#category-a)&nbsp;
      [How should so-called "Weak Copyleft" Licenses be handled](http://www.apache.org/legal/resolved.html#category-b)

- The dependency shall be **available in PyPi**.
    - Why?
        - Build reproducibility. Any repository server we use, must still run in future from now.
        - Build speed. More repositories slow down the build.
        - Build reliability. A repository server that is temporarily down can break builds.
    
- **Do not release the dependency yourself** (by building it from source).
    - Why? Because it's not an official release, by the official release guys.
        - A release must be 100% reproducible.
        - A release must be reliable (sometimes the release person does specific things you might not reproduce).

- **The sources are publicly available**
    - We may need to rebuild the dependency from sources ourselves in future. This may be in the rare case when
      the dependency is no longer maintained, but we need to fix a specific CVE there.
    - Make sure the dependency's pom.xml contains link to the source repository (`scm` tag).

- The dependency needs to use **reasonable build system**
    - Since we may need to rebuild the dependency from sources, we also need to make sure it is easily buildable.
      Maven or Gradle are acceptable as build systems.

- Only use dependencies with **an active community**.
    - Check for activity in the last year through [Open Hub](https://www.openhub.net).

- Less is more: **less dependencies is better**. Bloat is bad.
    - Try to use existing dependencies if the functionality is available in those dependencies
        - For example: use Apache Commons Math instead of Colt if Apache Commons Math is already a dependency

There are currently a few dependencies which violate some of these rules. They should be properly commented with a
warning and explaining why are needed
If you want to add a dependency that violates any of the rules above, get approval from the project leads.

### Tests and Documentation

Don't forget to include tests in your pull requests, and documentation (reference documentation, ...). 
Guides and reference documentation should be submitted to the [Python TrustyAI examples repository](https://github.com/trustyai-explainability/trustyai-explainability-python-examples).
If you are contributing a new feature, we strongly advise submitting an example.

### Code Reviews and Continuous Integration

All submissions, including those by project members, need to be reviewed by others before being merged. Our CI, GitHub Actions, should successfully execute your PR, marking the GitHub check as green.

## Feature Proposals

If you would like to see some feature in Python TrustyAI, just open a feature request and tell us what you would like to see.
Alternatively, you propose it during the [TrustyAI community meeting](https://github.com/trustyai-explainability/community).

Great feature proposals should include a short **Description** of the feature, the **Motivation** that makes that feature necessary and the **Goals** that are achieved by realizing it. If the feature is deemed worthy, then an Epic will be created.

## The small print

This project is an open source project, please act responsibly, be nice, polite and enjoy!

