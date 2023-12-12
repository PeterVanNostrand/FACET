# D3 SVG Generating Prototype

Instructions for using the first Explanation UI created from the code in the `/visualization/` directory

## Running the Visualization

To launch the explanation visualization tool do the following.

### Generate Explanations for the visualization

To generate explanations for the viz tool execute the following command

`python main.py --expr simple --method FACETIndex --ds loans`

You should see a set of `.JSON` files be created in the path `./visualizations/data`

### Install Node

Our user interface is built using JavaScript and D3 across several files and uses Node Package Manager to install and build all dependencies. To access this [install Node.js](https://nodejs.org/en/download) for your platform and add it to your path. When using default settings on Windows Node will install to `C:\Program Files\nodejs`. Be sure to check `Automatically install the necessary tools` during installation. Node.js version 18.16.0 and npm version 9.5.1 were used for development.

### Build the Project

In the directory `./visualizations` run `npm install package.json` and `npm run build`. This will create a `bundle.js` file which is linked to the `index.html` webpage we will access. You must rebuild the project after changing any relevant code.

If you're using VSCode this process can be automated using the [Trigger Task on Save](https://marketplace.visualstudio.com/items?itemName=Gruntfuggly.triggertaskonsave) extension. Once install configure a build task via the VSCode UI or by creating a `./vscode/tasks.json` file with the following contents.

```json
"version": "2.0.0",
    "tasks": [
        {
            "type": "npm",
            "script": "build",
            "path": "visualization",
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "problemMatcher": [],
            "label": "npm build",
            "detail": "rollup -c"
        }
    ]
```

Then add the following to your VSCode `settings.json` file (I suggest adding this only to the workspace settings file)

```json
"triggerTaskOnSave.on": true,
"triggerTaskOnSave.selectedTask": "npm build",
"triggerTaskOnSave.showBusyIndicator": true,
"triggerTaskOnSave.tasks": {
    "build": [
        "visualization/*.js",
        "*.js"
    ]
}
```

Now any time you save a `.js` file, Trigger Task on Save will execute the build action and VSCode will reload the rebuilt files. You can also trigger a build at any time using the VSCode shortcut `Ctrl + Shift + B` or command palette.

### Launch a Web Server

Due to security reasons for your browser to load the UI a webserver must serve the necessary files.

In the base project directory run `python -m http.server` then navigate to `http://localhost:XXXX/visualization/` in your browser, where `XXXX` is the port number provided by python. By default this should be port 8000.

You should see the following display ![/figures/ui_screenshot.jpg](/figures/ui_screenshot.jpg).

Alternatively if using VSCode with the [LiveServer Extension](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer) you can open `index.html` in VSCode and push the `Go Live` button on the bottom right side of the bottom bar.
