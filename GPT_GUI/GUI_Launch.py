import json
import logging
import tkinter as tk
from tkinter import messagebox
import sys
from tkinter import ttk, filedialog

# Check env has freqtrade path set

from freqtrade.main import main as freqtrade_main
from ttkthemes import ThemedTk

# from gui_dark_theme import create_dark_theme

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# TODO - When optional widgets are not turned on the data still populates in the config file. It should be removed to avoid conflicts with the different param requirements for each type of AI model
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None

    def show_tooltip(self):
        x = self.widget.winfo_pointerx() + 50
        y = self.widget.winfo_pointery() - 50

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = ttk.Label(self.tooltip, text=self.text, background="#ffffe0", relief="solid", borderwidth=1,
                          wraplength=250)
        label.pack()

    def hide_tooltip(self):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

def is_numeric_dataType(widget):
    return hasattr(widget, "dataType") and widget.dataType in ("IntVar", "DoubleVar")
def is_list_dataType(widget):
    return hasattr(widget, "dataType") and widget.dataType == "ListVar"

def get_widget_value(widget):
    if isinstance(widget, CustomEntry):
        return widget.get()
    elif isinstance(widget, CustomCheckbox):
        return widget.var.get()
    elif isinstance(widget, CustomListbox):
        return widget.get_selected_indices()
    elif isinstance(widget, CustomValueList):
        return widget.get_values()
    elif isinstance(widget, CustomOptionMenu):
        return widget.var.get()
    else:
        raise TypeError(f"Unsupported widget type: {type(widget)}")

class CustomEntry(ttk.Entry):
    def __init__(self, parent, label, row, column, tooltip_text, dataType, purpose, name, dictType=None, min_value=None,
                 max_value=None, possible_values=None, required=True, enabled=True):
        super().__init__(parent)
        self.label = label
        self.parent = parent
        self.dataType = dataType
        self.purpose = purpose
        self.min_value = min_value
        self.max_value = max_value
        self.possible_values = possible_values
        self.name = name
        self.dictType = dictType if dictType is not None else False

        ttk.Label(parent, text=label).grid(row=row, column=column, sticky='w')
        self.grid(row=row, column=column + 1)

        if not required:
            self.enable_var = tk.BooleanVar(value=enabled)
            enable_checkbutton = ttk.Checkbutton(parent, variable=self.enable_var, command=self.toggle_state)
            enable_checkbutton.grid(row=row, column=column + 2)
        else:
            self.enable_var = None

        tooltip = ToolTip(self, tooltip_text)
        self.bind("<Enter>", lambda e: tooltip.show_tooltip())
        self.bind("<Leave>", lambda e: tooltip.hide_tooltip())

    def toggle_state(self):
        if self.enable_var:
            if self.enable_var.get():
                self.configure(state='normal')
            else:
                self.configure(state='disabled')

class CustomCheckbox(ttk.Checkbutton):
    def __init__(self, parent, label, row, column, tooltip_text, purpose, name):
        self.var = tk.BooleanVar()
        self.label = label

        super().__init__(parent, text=label, variable=self.var)

        self.purpose = purpose
        self.name = name
        self.parent = parent

        self.grid(row=row, column=column, sticky='w')

        tooltip = ToolTip(self, tooltip_text)
        self.bind("<Enter>", lambda e: tooltip.show_tooltip())
        self.bind("<Leave>", lambda e: tooltip.hide_tooltip())

class CustomListbox(ttk.Frame):
    def __init__(self, parent, row, column, options, tooltip_text, label, purpose, name, defaultValue=None):
        super().__init__(parent)
        self.grid(row=row, column=column, padx=5, pady=5, sticky='nw')

        ttk.Label(self, text=label).pack()

        self.purpose = purpose
        self.name = name
        self.parent = parent
        self.options = options
        self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.label = label

        self.listbox = tk.Listbox(self, selectmode=tk.MULTIPLE, yscrollcommand=self.scrollbar.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        if defaultValue is not None:
            self.set_selected_indices(defaultValue)

        for option in options:
            self.listbox.insert(tk.END, option)

        self.scrollbar.config(command=self.listbox.yview)

        # Maintains highlighted selection when moving to another widget
        self.listbox.bind("<<ListboxSelect>>", self.on_listbox_select)
        self.listbox.configure(exportselection=False)

        if defaultValue is not None:
            for index in defaultValue:
                self.listbox.selection_set(index)

        # Set the tooltip for the listbox
        self.tooltip = ToolTip(self.listbox, tooltip_text)
        self.listbox.bind("<Enter>", lambda e: self.tooltip.show_tooltip())
        self.listbox.bind("<Leave>", lambda e: self.tooltip.hide_tooltip())

    def on_listbox_select(self, event):
        selected_indices = self.listbox.curselection()
        for index in selected_indices:
            self.listbox.selection_set(index)

    def get_selected_indices(self):
        return [self.listbox.index(item) for item in self.listbox.curselection()]

    def set_selected_indices(self, indices):
        self.listbox.selection_clear(0, tk.END)
        for index in indices:
            self.listbox.selection_set(index)

class CustomValueList(ttk.Frame):
    def __init__(self, parent, row, column, tooltip_text, label, purpose, dataType, name):
        super().__init__(parent)
        self.dataType = dataType
        self.label = label
        self.grid(row=row, column=column, padx=5, pady=5, sticky='nw')

        self.purpose = purpose
        self.name = name
        self.parent = parent

        ttk.Label(self, text=label).grid(row=0, column=0, sticky='w')

        self.entry = ttk.Entry(self)
        self.entry.grid(row=1, column=0, sticky='w')

        self.add_button = ttk.Button(self, text="Add", command=self.add_value)
        self.add_button.grid(row=1, column=1, padx=5)

        self.remove_button = ttk.Button(self, text='Remove', command=self.remove_selected_value)
        self.remove_button.grid(row=2, column=1, padx=2, pady=2)

        self.listbox = tk.Listbox(self)
        self.listbox.grid(row=2, column=0, columnspan=2, sticky='w')

        # Set the tooltip for the listbox
        self.tooltip = ToolTip(self.listbox, tooltip_text)
        self.listbox.bind("<Enter>", lambda e: self.tooltip.show_tooltip())
        self.listbox.bind("<Leave>", lambda e: self.tooltip.hide_tooltip())

    def update_values(self, values):
        self.listbox.delete(0, tk.END)
        for value in values:
            self.listbox.insert(tk.END, value)

    def add_value(self):
        value = self.entry.get().strip()
        if value:
            if self.dataType == 'int':
                try:
                    value = int(value)
                except ValueError:
                    messagebox.showerror("Invalid Input", f"{value} is not a valid integer")
                    return
            elif self.dataType == 'float':
                try:
                    value = float(value)
                except ValueError:
                    messagebox.showerror("Invalid Input", f"{value} is not a valid float")
                    return
            # Add more data type conversions if needed

            self.listbox.insert(tk.END, value)
            self.entry.delete(0, tk.END)

    def get_values(self):
        return [self.listbox.get(i) for i in range(self.listbox.size())]

    def remove_selected_value(self):
        selected_indices = self.listbox.curselection()
        if selected_indices:
            self.listbox.delete(selected_indices[0])

class CustomOptionMenu(ttk.Frame):
    def __init__(self, parent, label, row, column, tooltip_text, options, purpose, name, defaultValue=None):
        super().__init__(parent)
        self.grid(row=row, column=column, padx=5, pady=5, sticky='nw')

        self.purpose = purpose
        self.name = name
        self.parent = parent


        self.var = tk.StringVar()

        self.label = ttk.Label(self, text=label, width=25)
        self.label.grid(row=0, column=0, sticky='w')

        # Initialize the OptionMenu with the defaultValue
        self.option_menu = ttk.OptionMenu(self, self.var, defaultValue, *options)
        self.option_menu.config(width=25)
        self.option_menu.grid(row=0, column=1, sticky='w')

        tooltip = ToolTip(self.option_menu, tooltip_text)
        self.option_menu.bind("<Enter>", lambda e: tooltip.show_tooltip())
        self.option_menu.bind("<Leave>", lambda e: tooltip.hide_tooltip())

    def set_selected(self, value=None):
        if value is not None:
            self.var.set(value)

# Todo - add learning parameters like Eval metric and auc to xgboost  CHeck my eval methods on chatgpt and implement that info for tooltips
# TODO - Update AI parameters with some of the options available here
#   https://xgboost.readthedocs.io/en/stable/parameter.html
#   Skipped grow policy, monotone_constraints, interaction_constraints and process type to begin with
# todo - all the parameters in the "todo" section of this website are not yet added to my code. https://www.freqtrade.io/en/stable/configuration/
#   Also missing telegram, API, bot name, internals and lots of others
#   Also missing Hyperopt and Backtest specific parameters
class FreqTradeGUI:
    def __init__(self, root, debug=0):
        self.all_tab_widgets = []
        self.root = root
        self.debug = debug

        # set the window size and position
        window_width = 1500
        window_height = 850
        x_pos = 0
        y_pos = 0
        root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")

        self.root.title("FreqTrade GUI")

        self.setup_notebook()
        self.load_frame_config()
        # Initialize self.widget_config before calling self.load_widget_config()
        self.widget_config = {"entry_widgets": [], "checkboxes": [], "listboxes": [], "optionboxes": [], "valuelists": []}
        self.widget_config = self.load_widget_config()

        self.create_frames()
        self.create_widgets()
        self.update_widgets_from_file()

        if self.debug == 1:
            self.root.after(100, self.run_button_sequence)

    def setup_notebook(self):
        """Sets up the notebook (tab container) and its tabs."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill=tk.BOTH)

        self.launch_tab = self.create_scrollable_tab(self.notebook, "Launch")
        self.config_tab = self.create_scrollable_tab(self.notebook, "Config")
        self.ai_model_config_tab = self.create_scrollable_tab(self.notebook, "AI Parameters")

    def create_scrollable_tab(self, notebook, tab_name):
        """Create a scrollable tab with both vertical and horizontal scrollbars and add it to the notebook."""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=tab_name)
        # Get the background color of the ttk theme
        bg_color = self.root.tk.call("ttk::style", "lookup", "TFrame", "-background")
        # Apply the background color to the canvas
        canvas = tk.Canvas(tab, bg=bg_color)

        v_scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=v_scrollbar.set)
        h_scrollbar = ttk.Scrollbar(tab, orient=tk.HORIZONTAL, command=canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.configure(xscrollcommand=h_scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollable_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=scrollable_frame, anchor=tk.NW)

        def update_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox(tk.ALL))

        scrollable_frame.bind("<Configure>", update_scroll_region)

        return scrollable_frame

    def create_widgets(self):
        """Create widgets based on the configuration."""
        self.create_entry_widgets()
        self.create_checkbox_widgets()
        self.create_listbox_widgets()
        self.create_optionmenu_widgets()
        self.create_custom_valuelist_widgets()
        self.create_buttons()

    def create_entry_widgets(self):
        """Create entry widgets."""
        for entry_config in self.widget_config["entry_widgets"]:
            parent_frame = self.frames[entry_config["parent"]]
            entry_widget = CustomEntry(parent_frame, entry_config["label"], entry_config["row"], entry_config["column"],
                                       entry_config["tooltip_text"], entry_config["dataType"],
                                       dictType=entry_config.get("dictType"),
                                       min_value=entry_config.get("min_value"),
                                       max_value=entry_config.get("max_value"),
                                       possible_values=entry_config.get("possible_values"),
                                       required=entry_config.get("required", True),
                                       enabled=entry_config.get("enabled", True),
                                       purpose=entry_config.get("purpose"),
                                       name=entry_config.get("name"))
            entry_widget.grid(padx=5, pady=5)
            self.all_tab_widgets.append(entry_widget)

    def create_checkbox_widgets(self):
        """Create checkbox widgets."""
        for checkbox_config in self.widget_config.get("checkboxes", []):
            parent_name = checkbox_config["parent"]
            parent_frame = self.frames[parent_name]

            checkbox_widget = CustomCheckbox(parent_frame, checkbox_config["label"], checkbox_config["row"],
                                             checkbox_config["column"], checkbox_config["tooltip_text"],
                                             purpose=checkbox_config.get("purpose"),
                                             name=checkbox_config.get("name"))
            checkbox_widget.grid(padx=5, pady=5, sticky='nw')
            self.all_tab_widgets.append(checkbox_widget)

    def create_listbox_widgets(self):
        """Create listbox widgets."""
        for listbox_config in self.widget_config["listboxes"]:
            parent_frame = self.frames[listbox_config["parent"]]
            listbox_widget = CustomListbox(parent_frame, listbox_config["row"], listbox_config["column"],
                                           listbox_config["listbox_items"], listbox_config["tooltip_text"],
                                           listbox_config["label"],
                                           defaultValue=listbox_config.get("defaultValue"),
                                           purpose=listbox_config.get("purpose"), name=listbox_config.get("name")
                                           )
            listbox_widget.grid(padx=5, pady=5)
            self.all_tab_widgets.append(listbox_widget)

    def create_optionmenu_widgets(self):
        """Create OptionMenu widgets."""
        for optionbox_config in self.widget_config["optionboxes"]:
            parent_frame = self.frames[optionbox_config["parent"]]
            optionbox_widget = CustomOptionMenu(parent_frame, optionbox_config["label"], optionbox_config["row"],
                                                optionbox_config["column"], optionbox_config["tooltip_text"],
                                                optionbox_config["options"],
                                                defaultValue=optionbox_config.get("defaultValue"),
                                                purpose=optionbox_config.get("purpose"),
                                                name=optionbox_config.get("name"))
            optionbox_widget.grid(padx=5, pady=5)
            self.all_tab_widgets.append(optionbox_widget)

    def create_custom_valuelist_widgets(self):
        """Create CustomValueList widgets."""
        for valuelist_config in self.widget_config["valuelists"]:
            parent_frame = self.frames[valuelist_config["parent"]]
            valuelist_widget = CustomValueList(parent_frame, valuelist_config["row"], valuelist_config["column"],
                                               valuelist_config["tooltip_text"], valuelist_config["label"],
                                               purpose=valuelist_config.get("purpose"),
                                               dataType=valuelist_config.get("dataType"),
                                               name=valuelist_config.get("name")

                                               )
            valuelist_widget.grid(padx=5, pady=5)
            self.all_tab_widgets.append(valuelist_widget)

    def create_buttons(self):
        """Create buttons."""
        self.save_button = ttk.Button(self.root, text='Save To Specific File',
                                      command=self.save_button_process)
        self.save_button.pack(side='left', padx=5, pady=2)

        self.load_button = ttk.Button(self.root, text='Load', command=self.load_button_process)
        self.load_button.pack(side='left', padx=5, pady=2)

        self.update_defaults_button = ttk.Button(self.root, text='Update Defaults',
                                                 command=self.update_defaults_button_process)
        self.update_defaults_button.pack(side='left', padx=5, pady=2)

        self.run_button = ttk.Button(self.root, text='Run', command=self.run_button_sequence)
        self.run_button.pack(side='right', padx=5, pady=2)

    def load_frame_config(self):
        try:
            with open("gui_frame_config.json", "r") as f:
                self.frame_config = json.load(f)
        except FileNotFoundError:
            logging.error("Error: gui_frame_config.json not found. Using default frame configuration.")

    def create_frames(self):
        self.frames = {}

        for frame in self.frame_config["frames"]:
            name = frame["name"]
            label = frame["label"]
            parent = frame["parent"]
            generic = frame.get("generic", False)  # Get the generic attribute or default to False

            grid_options = {k: v for k, v in frame.items() if k in ("row", "column", "rowspan", "columnspan")}

            if parent == "launch_tab":
                parent = self.launch_tab
            elif parent == 'config_tab':
                parent = self.config_tab
            elif parent == 'ai_model_config_tab':
                parent = self.ai_model_config_tab
            else:
                parent = self.frames[parent]

            frame = ttk.LabelFrame(parent, text=label)
            frame.grid(**grid_options, sticky='nsew', padx=5, pady=5)
            frame.name = name  # Store the name attribute in the frame object
            frame.parent = parent
            frame.generic = generic  # Store the generic attribute in the frame object
            # store the created frame object in the self.frames dictionary with the associated name as the key
            self.frames[name] = frame

    def load_widget_config(self, file_path=None):
        try:
            if not file_path:
                file_path = "gui_widget_previous_run.json"
            with open(file_path, "r") as f:
                loaded_values = json.load(f)

            # Update widget_config with values from loaded_values
            for widget_type, widget_list in loaded_values.items():
                for config in widget_list:
                    for widget_config in self.widget_config[widget_type]:
                        if widget_config["label"] == config["label"]:
                            widget_config["defaultValue"] = config["defaultValue"]
                            # Load the enabled attribute
                            if "enabled" in config:
                                widget_config["enabled"] = config["enabled"]

        except FileNotFoundError:
            logging.error("Error: freqtrade\\gui_widget_previous_run.json not found. Using default previous values.")
            loaded_values = {}
        except json.JSONDecodeError:
            logging.error("Error: Failed to decode gui_widget_previous_run.json. Using default previous values.")
            loaded_values = {}

        # Log the counts for each widget type
        logging.info(f"Entry widgets: {len(loaded_values.get('entry_widgets', []))}")
        logging.info(f"Checkboxes: {len(loaded_values.get('checkboxes', []))}")
        logging.info(f"Listboxes: {len(loaded_values.get('listboxes', []))}")
        logging.info(f"Optionboxes: {len(loaded_values.get('optionboxes', []))}")
        logging.info(f"CustomValueList widgets: {len(loaded_values.get('valuelists', []))}")

        # Update widgets with values from the widget_config
        self.update_widgets_from_file()

        return loaded_values

    def update_widgets_from_file(self):
        for widget in self.all_tab_widgets:
            if isinstance(widget, CustomEntry):
                for entry_config in self.widget_config["entry_widgets"]:
                    if entry_config["label"] == widget.label:
                        widget.delete(0, tk.END)
                        if entry_config["defaultValue"] is not None:
                            widget.insert(0, entry_config["defaultValue"])
                        # Apply the enabled attribute, if it exists
                        if "enabled" in entry_config:
                            if widget.enable_var:
                                widget.enable_var.set(entry_config["enabled"])
                                widget.toggle_state()
                        else:
                            # Default behavior if the enabled attribute is not present
                            widget.configure(state=tk.NORMAL)
            elif isinstance(widget, CustomCheckbox):
                for checkbox_config in self.widget_config["checkboxes"]:
                    if checkbox_config["label"] == widget["text"]:
                        widget.var.set(checkbox_config["defaultValue"])
            elif isinstance(widget, CustomListbox):
                for listbox_config in self.widget_config["listboxes"]:
                    if listbox_config["label"] == widget.label:
                        widget.set_selected_indices(listbox_config["defaultValue"])
            elif isinstance(widget, CustomOptionMenu):
                for optionbox_config in self.widget_config["optionboxes"]:
                    if optionbox_config["label"] == widget.label:
                        widget.set_selected(optionbox_config["defaultValue"])
            elif isinstance(widget, CustomValueList):
                for valuelist_config in self.widget_config["valuelists"]:
                    if valuelist_config["label"] == widget.label:
                        widget.update_values(valuelist_config["defaultValue"])

    def update_file_from_widgets(self):
        for widget in self.all_tab_widgets:
            if isinstance(widget, CustomEntry):
                for entry_config in self.widget_config["entry_widgets"]:
                    if entry_config["name"] == widget.name:
                        value = get_widget_value(widget)
                        if value == '':
                            entry_config["defaultValue"] = None
                        else:
                            if is_numeric_dataType(widget):
                                if widget.dataType == "DoubleVar":
                                    entry_config["defaultValue"] = float(value)
                                elif widget.dataType == "IntVar":
                                    entry_config["defaultValue"] = int(value)
                            else:
                                entry_config["defaultValue"] = value
                        # Save the enabled attribute
                        if widget.enable_var:
                            entry_config["enabled"] = widget.enable_var.get()
            elif isinstance(widget, CustomOptionMenu):
                for optionbox_config in self.widget_config["optionboxes"]:
                    if optionbox_config["name"] == widget.name:
                        optionbox_config["defaultValue"] = get_widget_value(widget)
            elif isinstance(widget, CustomValueList):
                for valuelist_config in self.widget_config["valuelists"]:
                    if valuelist_config["name"] == widget.name:
                        values = get_widget_value(widget)
                        if is_numeric_dataType(widget):
                            if widget.dataType == "DoubleVar":
                                values = [float(value) for value in values]
                            elif widget.dataType == "IntVar":
                                values = [int(value) for value in values]
                        valuelist_config["defaultValue"] = values
            elif isinstance(widget, CustomListbox):
                for listbox_config in self.widget_config["listboxes"]:
                    if listbox_config["name"] == widget.name:
                        listbox_config["defaultValue"] = get_widget_value(widget)
            elif isinstance(widget, CustomCheckbox):
                for checkbox_config in self.widget_config["checkboxes"]:
                    if checkbox_config["name"] == widget.name:
                        checkbox_config["defaultValue"] = get_widget_value(widget)

    def load_button_process(self):
        file_path = filedialog.askopenfilename(title="Select a file",
                                               filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not file_path:
            return

        self.load_widget_config(file_path)

    def save_button_process(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json")
        if not file_path:
            return
        self.update_file_from_widgets()
        with open(file_path, "w") as f:
            json.dump(self.widget_config, f, indent=4)

    def update_defaults_button_process(self):
        self.update_file_from_widgets()
        with open("gui_widget_default_configs.json", "w") as f:
            json.dump(self.widget_config, f, indent=4)

    # run is used to set the order of functions required to be executed when the script is run
    def run_button_sequence(self):
        self.update_defaults_button_process()
        self.overwrite_configuration_file()
        self.launch_freqtrade()
        self.root.destroy()  # Close the GUI

    # overwrite_configuration_file is used to set the config file used by freqtrade
    def overwrite_configuration_file(self):

        def is_generic_frame(frame_name, frames):
            frame = frames.get(frame_name)
            if frame and hasattr(frame, 'generic'):
                return frame.generic
            return False

        def get_parent_hierarchy(widget, frames):
            hierarchy = []
            full_hierarchy = []  # This will include all parents, even generic ones
            parent = widget.master
            while parent != self.root and parent != self.notebook:
                parent_name_list = [name for name, frame in frames.items() if frame == parent]
                if parent_name_list:
                    parent_name = parent_name_list[0]
                    full_hierarchy.append(parent_name)  # Add every parent to the full_hierarchy
                    # Add the current parent to the hierarchy if it's not generic and not "config_standard"
                    if parent_name != "config_standard" and not is_generic_frame(parent_name, frames):
                        hierarchy.append(parent_name)
                    # If the parent is generic, move up the hierarchy to its parent
                    if is_generic_frame(parent_name, frames):
                        parent = parent.master
                        continue
                    # Stop adding parents to the hierarchy when encountering "config_standard"
                    if parent_name == "config_standard":
                        break
                parent = parent.master
            return hierarchy[::-1], full_hierarchy[::-1]

        def set_nested_value(dictionary, keys, value):
            if not keys:
                dictionary[keys[-1]] = value
            else:
                for key in keys[:-1]:
                    if key != 'pairlists':
                        dictionary = dictionary.setdefault(key, {})
                if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
                    try:
                        dictionary[keys[-1]] = json.loads(value)
                    except json.JSONDecodeError:
                        logging.error(f"Error: Failed to decode the string '{value}' as a dictionary.")
                        dictionary[keys[-1]] = value
                else:
                    # Check if the widget's parent is "pairlists"
                    if len(keys) >= 2 and keys[-2] == "pairlists":
                        # If the parent is "pairlists", append the value to the list in the dictionary
                        pairlist_dict = {keys[-1]: value}
                        if keys[-2] not in dictionary:
                            dictionary[keys[-2]] = [pairlist_dict]
                        else:
                            dictionary[keys[-2]].append(pairlist_dict)
                    else:
                        dictionary[keys[-1]] = value

        # Deals with widgets that contain child dictionaries of values
        def has_nested_children(widget):
            if not isinstance(widget, tk.Widget):
                return False
            children = widget.winfo_children()
            if not children:
                return False
            for child in children:
                if isinstance(child, tk.Widget) or has_nested_children(child):
                    return True
            return False

        def widget_sort_key(widget):
            return has_nested_children(widget)

        def should_exclude_widget(widget, frames, freqaimodel_value):
            parent_hierarchy, full_hierarchy = get_parent_hierarchy(widget, frames)
            if 'model_training_parameters' in parent_hierarchy:
                model_name_prefix = freqaimodel_value.split("Regressor", 1)[0].split("Classifier", 1)[0]
                return model_name_prefix.lower() not in full_hierarchy  # use full_hierarchy here
            return False

        # find the defaultValue of the --freqaimodel widget before the loop starts
        freqaimodel_value = None
        for widget in self.all_tab_widgets:
            if hasattr(widget, "name") and widget.name == "--freqaimodel":
                freqaimodel_value = get_widget_value(widget)
                break

        config_data = {}
        sorted_widgets = sorted(self.all_tab_widgets, key=widget_sort_key)
        for widget in sorted_widgets:
            if should_exclude_widget(widget, self.frames, freqaimodel_value):
                continue
            # Skip CustomEntry widgets that have enabled = false
            if isinstance(widget, CustomEntry) and widget.enable_var and not widget.enable_var.get():
                continue
            if hasattr(widget, "purpose") and widget.purpose == "config":
                hierarchy, full_hierarchy = get_parent_hierarchy(widget, self.frames)
                # In the for loop that iterates over self.all_tab_widgets
                if isinstance(widget, CustomEntry):
                    entry_value = get_widget_value(widget)
                    if is_numeric_dataType(widget):
                        entry_value = int(entry_value) if widget.dataType == "IntVar" else float(entry_value)
                    # If the widget has a "dictType" attribute and it is True, convert the entry value to a string representing a dictionary
                    elif getattr(widget, 'dictType', False) and isinstance(entry_value, dict):
                        entry_value = json.dumps(entry_value)
                    set_nested_value(config_data, hierarchy + [widget.name], entry_value)
                elif isinstance(widget, CustomCheckbox):
                    set_nested_value(config_data, hierarchy + [widget.name], get_widget_value(widget))
                elif isinstance(widget, CustomListbox):
                    selected_indices = get_widget_value(widget)
                    selected_values = [widget.listbox.get(i) for i in selected_indices]
                    set_nested_value(config_data, hierarchy + [widget.name], selected_values)
                elif isinstance(widget, CustomValueList):
                    set_nested_value(config_data, hierarchy + [widget.name], get_widget_value(widget))
                elif isinstance(widget, CustomOptionMenu):
                    set_nested_value(config_data, hierarchy + [widget.name], get_widget_value(widget))

        with open("C:\\FreqTradeStuff\\freqtrade\\user_data\\configs\\gui_generated.json", "w") as f:
            json.dump(config_data, f, indent=4)

    def launch_freqtrade(self):
        # Initialise with the value of widgets
        mode = None
        start_date = None
        end_date = None
        for widget in self.all_tab_widgets:
            if widget.name == "mode":
                mode = get_widget_value(widget)
            elif widget.name == "start_date":
                start_date = get_widget_value(widget)
            elif widget.name == "end_date":
                end_date = get_widget_value(widget)


        def is_inside_mode_specific_settings(widget):
            parent = widget.master
            mode_specific_settings_frame = self.frames["mode_specific_settings"]
            while parent != self.root and parent != self.notebook:
                if parent.master == mode_specific_settings_frame:
                    return True
                if parent.name == "bot_start":
                    break
                parent = parent.master
            return False

        # Collect the widget values with purpose attribute = "launch"
        launch_params = []
        for widget in self.all_tab_widgets:
            if getattr(widget, "purpose", None) == "launch":
                # Check if the widget is inside 'mode_specific_settings'
                if is_inside_mode_specific_settings(widget):
                    # If the widget's parent frame name does not match the selected mode value, skip this widget
                    if widget.master.name != mode:
                        continue

                value = get_widget_value(widget)
                if value:
                    if widget.name == "mode" or widget.name == "end_date":
                        continue
                    elif widget.name == "--config":
                        # C:\FreqTradeStuff\freqtrade\user_data\new_test_config.json
                        launch_params.append(widget.name + " C:\\FreqTradeStuff\\freqtrade\\user_data\\configs\\" + value)
                        launch_params.append(widget.name + " C:\\Users\\engin\\OneDrive\\Desktop\\config_private.json")
                    elif widget.name == "start_date":
                        timerange_arg = "--timerange " + start_date + "-" + end_date
                        launch_params.append(timerange_arg)
                    elif isinstance(widget, CustomListbox):
                        # handle CustomListbox widget
                        value_list = widget.options  # directly access the list of options
                        selected_options = [value_list[index] for index in value]
                        pairs_string = " ".join(
                            [f"{option}" for option in selected_options])  # Remove the quotes around {option}
                        launch_params.append(
                            f"{widget.name} {pairs_string}")  # Remove the brackets around {pairs_string}
                    else:
                        launch_params.append(widget.name + " " + value)

        # Construct the CLI string
        if mode == "download-data":
            cli_string = f"freqtrade {mode}"
        else:
            cli_string = f"freqtrade {mode} --strategy-path C:\\FreqTradeStuff\\freqtrade\\user_data\\strategies"

        for param in launch_params:
            if mode == "download-data":
                if any(param.startswith(x) for x in
                       ("--exchange", "--pairs", "--timeframes", "--trading-mode", "--timerange")):
                    cli_string += f" {param}"

            # Don't load a model if not required
            elif param == '--freqaimodel DisableFreqAI':
                pass
            else:
                cli_string += f" {param}"


        # TODO - Debug hard code to be over written
        #  Issues found - Missing :USDT at end of correlated coins meant I had NAN columns for a coin with no history "BTC/USDT"
        #  Also, not including enough start up candles
        #  Both result in a warning about training data size
        # cli_string = """freqtrade hyperopt --strategy BasicAI --spaces buy sell stoploss trailing --timerange 20230503-20230505 --pairs BTC/USDT:USDT --hyperopt-loss OnlyProfitHyperOptLoss --freqaimodel CatboostClassifierMultiTarget --strategy-path C:\\FreqTradeStuff\\freqtrade\\user_data\\strategies --config C:\\FreqTradeStuff\\freqtrade\\user_data\\basicAI_config.json  --config C:\\Users\\engin\\OneDrive\\Desktop\\config_private.json"""
        # # hard coded print all for hyperopt outputs
        # if mode == "hyperopt":
        #    cli_string += f" --print-all"
        # Call the freqtrade main file using the constructed CLI string
        logging.info(f"Executing:\n\n################################## Sending Following String to CLI ##################################"
                     f"\n{cli_string}\n"
                     f"################################## ############################### ##################################\n")
        cli_args = cli_string.split()[1:]  # Exclude the 'freqtrade' command itself


        # debug
        # print(freqtrade_main)
        # print(type(freqtrade_main))
        freqtrade_main(cli_args)

def main():
    root = ThemedTk(theme="blue")
    debug = 0  # Default value for debug

    # Check if a debug argument is passed and set debug accordingly
    if len(sys.argv) > 1 and sys.argv[1] == '1':
        print(f"\nDebug Mode enabled by sys.argv=1, this will automatically trigger the UI to run with previous conditions\n")
        print(f"\nSearch the gui wdiget default and previous json files to manually edit any params that seem stuck\n")
        debug = 1

    app = FreqTradeGUI(root, debug=debug)
    root.update()  # Update the GUI to compute the required size for content
    root.minsize(root.winfo_width(), root.winfo_height())  # Set the minimum size to the required size
    root.mainloop()

if __name__ == "__main__":
    main()
