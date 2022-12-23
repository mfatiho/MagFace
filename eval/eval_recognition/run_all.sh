INDEXES_FILES="indexes/*"

for f in $INDEXES_FILES
do
  for d in lfw cfp agedb
  do
    #echo "\n\nProcessing '$(basename -- $f)' for dataset '$d'"
    python eval_1v1_with_select_feats.py --feat-list ./features/magface_iresnet100/${d}_official.list --pair-list data/$d/pair.list --feat-indexes "$(basename -- $f)"
  done
done

# python3 eval_1v1_with_select_feats.py --feat-list ./features/magface_iresnet100/cfp_official.list --pair-list data/cfp/pair.list --feat-indexes agedb_official_genetic_indexes.npy
