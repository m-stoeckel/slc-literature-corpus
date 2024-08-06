push TARGET="/storage/projects/stoeckel/syntactic-language-change/literature-corpus/":
    rsync -rlhcv --info=Progress2 --delete-after --copy-links --exclude='__pycache__' --exclude='.git' src {{TARGET}}

push-gh:
    @just push "fuchs:/scratch/fuchs/agmisc/stoeckel/projects/slc/"

push-rbi:
    @just push "adonis:~/projects/slc/"

txt:
    rsync -rh -m --info=Progress2 \
        --exclude '/[[:alpha:]]*' \
        --exclude '*[[:alpha:]]/' \
        --include '[123456789]*/' \
        --include '**/[0123456789]*/' \
        --include '[0123456789]*-[08].zip' \
        --exclude '*.*' rsync://rsync.mirrorservice.org/gutenberg.org/ data/mirror/num-0_8-zip/

html +FLAGS='':
    rsync -rh -m {{FLAGS}} --info=Progress2 \
        --exclude '/[[:alpha:]]*' \
        --exclude '*[[:alpha:]]/' \
        --include '[123456789]*/' \
        --include '**/[0123456789]*/' \
        --include '[0123456789]*-h.zip' \
        --exclude '*.*' rsync://rsync.mirrorservice.org/gutenberg.org/ data/mirror/num-h-zip/
