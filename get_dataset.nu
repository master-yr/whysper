#!/usr/bin/env nu

# Fetch with progress
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzf dev-clean.tar.gz

# Organize
mkdir dataset
glob "**/*.trans.txt" | each { |f|
    open $f | lines | each { |l|
        let p = ($l | parse "{id} {text}" | first)
        let flac = $"(($f | path dirname))/($p.id).flac"
        if ($flac | path exists) {
            # Convert FLAC to WAV instead of copying
            ffmpeg -i $flac -ar 16000 -ac 1 -y $"dataset/($p.id).wav" err> /dev/null
            $p.text | save $"dataset/($p.id).txt"
        }
    }
}

# Cleanup
rm -rf LibriSpeech dev-clean.tar.gz
ls dataset | where name =~ '\.wav$' | length | print $"✅ Done! ($in) files ready"

