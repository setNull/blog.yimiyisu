#/bin/bash
echo -e "\033[0;32mDeploying updates to GitHub...\033[0m"
des="public"
msg="rebuilding site `date`"
if [ $# -eq 1  ]
    then msg="$1"
fi

git add -A
git commit -m "$msg"
git push origin master

# Build the project.
hugo # if using a theme, replace by `hugo -t <yourtheme>`
hugo-algolia --config algolia.yaml -s

#cp -r public/* $des
# Go To Public folder
cd $des


git add -A
git commit -m "$msg"
git push origin master

