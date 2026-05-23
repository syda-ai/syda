Run the syda marketing release workflow.

The marketing CLI lives at:
  ~/Library/CloudStorage/GoogleDrive-ramkumar2606@gmail.com/My Drive/projects/syda/marketing/

Always cd to that folder first and use its venv:
  cd ~/Library/CloudStorage/GoogleDrive-ramkumar2606@gmail.com/My\ Drive/projects/syda/marketing

Python binary: .venv/bin/python
CLI entry:     release.py

## Steps

1. Ask the user for the version number if not already provided (e.g. "0.0.6").

2. Run prepare to generate content from CHANGELOG:
   .venv/bin/python release.py prepare <version>

3. Run review so the user can read the generated content:
   .venv/bin/python release.py review <version>

4. Tell the user they can now edit the files in content/v<version>/ before posting, and ask which platforms to post to: reddit, linkedin, twitter, or all.

5. Run post with the chosen platform:
   .venv/bin/python release.py post <version> --platform <choice>
   (or omit --platform to post to all)

Use --dry-run if the user wants to preview without actually posting.
