#!/bin/bash
# Packages the final model into the organizer's expected directory

echo "Packaging final submission..."

if [ ! -f "final_model.pkl" ]; then
    echo "ERROR: submission/final_model.pkl not found!"
    echo "Did your server finish all rounds successfully?"
    exit 1
fi

# The organizer's setup_infra.sh creates this directory for collection
TARGET_DIR="/home/hackadmin/submission"

if [ -d "$TARGET_DIR" ]; then
    cp final_model.pkl "$TARGET_DIR/final_model.pkl"
    echo "Success! Model has been copied to $TARGET_DIR."
    echo "The organizers will collect it automatically at the deadline."
else
    echo "Target directory $TARGET_DIR does not exist."
    echo "Are you running this on the Team VM?"
fi