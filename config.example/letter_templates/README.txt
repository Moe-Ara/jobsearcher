COVER-LETTER TEMPLATES
======================

Put human-written cover letters in this folder: your own past letters that worked, or good examples you found.
The letter writer copies their STRUCTURE and VOICE:
  - number and order of paragraphs, and what each paragraph does
  - how the letter opens and closes
  - paragraph and sentence lengths, tone and formality
It NEVER uses their facts. Names, employers, degrees, numbers and technologies always come from config/profile.yaml.
If something from a template slips into a letter anyway, the fact check flags it.

Easiest: use the "Templates" page in the app (paste, edit, delete).
Or drop files here: one letter per file, .md or .txt, UTF-8. The language is detected automatically.
Optional header for more control:

---
title: Software developer, direct style
language: de          # de | en
tags: [testing, java, automotive]   # helps pick the best template for a job
notes: my letter to X, got an interview
---
Sehr geehrte Frau ...,

...

Mit freundlichen Grüßen
...

Tips:
- Paste the WHOLE letter, from salutation to closing. Address blocks and dates above the salutation are ignored.
  So is the name under the closing.
- 3-6 good letters per language work better than one. For each letter, the best-matching template sets the
  structure; a second one adds style (preferences.yaml: writing.templates_per_letter).
- The letter length follows your templates (their body length, plus or minus 15 %).
- Files whose name starts with README are ignored.
