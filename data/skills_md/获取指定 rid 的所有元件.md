使用getComponentsByRid(rid)获取模型中获取rid =model/CloudPSS/_newBreaker_3p的所有元件
        ```'codes':```python
model = cloudpss_substation_Model
components = model.getComponentsByRid(rid='model/CloudPSS/_newBreaker_3p')
print(components)
```
'output':```python
{
    'component_new_breaker_3_p_48': <cloudpss.model.implements.component.Componentobjectat0x000002591F764C70>,
    'component_new_breaker_3_p_49': <cloudpss.model.implements.component.Componentobjectat0x000002591F764CA0>,
    'component_new_breaker_3_p_50': <cloudpss.model.implements.component.Componentobjectat0x000002591F764CD0>,
    'component_new_breaker_3_p_53': <cloudpss.model.implements.component.Componentobjectat0x000002591F764D00>,
    'component_new_breaker_3_p_54': <cloudpss.model.implements.component.Componentobjectat0x000002591F764D30>,
    'component_new_breaker_3_p_55': <cloudpss.model.implements.component.Componentobjectat0x000002591F764D60>,
    'component_new_breaker_3_p_56': <cloudpss.model.implements.component.Componentobjectat0x000002591F764D90>,
    'component_new_breaker_3_p_57': <cloudpss.model.implements.component.Componentobjectat0x000002591F764DC0>,
    'component_new_breaker_3_p_58': <cloudpss.model.implements.component.Componentobjectat0x000002591F764DF0>,
    'component_new_breaker_3_p_59': <cloudpss.model.implements.component.Componentobjectat0x000002591F764E20>,
    'component_new_breaker_3_p_6': <cloudpss.model.implements.component.Componentobjectat0x000002591F764E50>,
    'component_new_breaker_3_p_60': <cloudpss.model.implements.component.Componentobjectat0x000002591F764E80>,
    'component_new_breaker_3_p_61': <cloudpss.model.implements.component.Componentobjectat0x000002591F764EB0>,
    'component_new_breaker_3_p_62': <cloudpss.model.implements.component.Componentobjectat0x000002591F764EE0>,
    'component_new_breaker_3_p_63': <cloudpss.model.implements.component.Componentobjectat0x000002591F764F10>,
    'component_new_breaker_3_p_64': <cloudpss.model.implements.component.Componentobjectat0x000002591F764F40>,
    'component_new_breaker_3_p_65': <cloudpss.model.implements.component.Componentobjectat0x000002591F764F70>,
    'component_new_breaker_3_p_66': <cloudpss.model.implements.component.Componentobjectat0x000002591F764FA0>,
    'component_new_breaker_3_p_67': <cloudpss.model.implements.component.Componentobjectat0x000002591F764FD0>,
    'component_new_breaker_3_p_69': <cloudpss.model.implements.component.Componentobjectat0x000002591F765000>,
    'component_new_breaker_3_p_7': <cloudpss.model.implements.component.Componentobjectat0x000002591F765030>,
    'component_new_breaker_3_p_8': <cloudpss.model.implements.component.Componentobjectat0x000002591F765060>,
    'component_new_breaker_3_p_9': <cloudpss.model.implements.component.Componentobjectat0x000002591F765090>
}
```
        