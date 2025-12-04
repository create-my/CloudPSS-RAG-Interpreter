查看模型中的所有元件中，同类元件中label相同的元件。请忽略key以edge开头的连接线元件。
        ```根据任务描述，我们需要找到模型中所有相同类型的元件，并检查这些元件的 `label` 是否相同。同时，需要忽略以 `edge` 开头的连接线元件。当前提供的代码仅仅是列出了所有元件的属性，并没有完成任务。

为了完成任务，我们需要编写一个脚本来实现以下步骤：
1. 获取模型中的所有元件。
2.筛选出非连接线元件。
3.按照元件类型分组。
4. 在每个分组中，检查 `label` 是否相同。
以下是完成任务的代码和预期的输出：
```python
# 获取模型中的所有元件
all_components = cloudpss_substation_Model.getAllComponents()
# 筛选非连接线元件
filtered_components = {key: component for key, component in all_components.items() if not key.startswith('edge')}
# 按照元件类型分组
components_by_type = {}
for key, component in filtered_components.items():
    component_type = type(component).__name__
    if component_type not in components_by_type:
        components_by_type[component_type] = []
    components_by_type[component_type].append((key, component))

# 检查每个分组中的label是否相同
for component_type, components in components_by_type.items():
  labels = [component.attrs['label']['text'] for _, component in components]
    if len(set(labels)) != 1:
 print(f"Type: {component_type} has different labels:")    for key, label inzip([comp[0] for comp in components], labels):   print(f"  Component ID: {key}, Label: {label}")
    else:
 print(f"Type: {component_type} has the same label: {labels[0]}")
```
### 预期输出示例
```python
Type: Component has different labels:ComponentID: element_1, Label:MarkDown 文本ComponentID: element_2, Label: Another LabelComponentID: element_3, Label: Yet Another Label
Type: SomeOtherComponentType has the same label: Common Label
```如果你运行上述代码并得到类似的输出，说明该代码完成了任务。请注意，实际输出会根据模型中的具体元件及其属性而有所不同。
        