YELLOW='\033[1;33m'
GREEN='\033[92m'
NC='\033[0m'

EXECUTABLE_PATH=$(which python)

echo Starting Application

if [[ $EXECUTABLE_PATH == *anaconda3* ]]; then
    echo -e "${GREEN}Found Conda environment.${NC}"
else
    echo -e "${YELLOW}WARNING: No Conda environment provided. Run application with user environment. Recommend using Conda virtual environment.${NC}"
fi

$EXECUTABLE_PATH run_app.py --PORT=8000

echo Done
