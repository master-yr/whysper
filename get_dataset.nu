#!/usr/bin/env nu

# Download and extract Google Speech Commands Dataset (single words)
# wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir ./dataset
tar -xzf speech_commands_v0.02.tar.gz -C dataset
# cp -r ./dataset_unprocessed ./dataset

# # Organize files: move WAVs + create matching TXT files
# ls dataset/**/*.wav | each { |f|
#     let word = $f.name | path basename | split row '_' | get 0;
#     let new_dir = $"dataset/($word)"

#     mkdir $new_dir
#     mv $f.name $"($new_dir)/($f.name | path basename)"

#     # Create label file
#     $word | save -f $"($new_dir)/($f.name | path basename | str replace '.wav' '.txt')"
# }

# Cleanup
# rm speech_commands_v0.02.tar.gz
print $"✅ Done! (ls dataset/**/*.wav | length) files organized"
