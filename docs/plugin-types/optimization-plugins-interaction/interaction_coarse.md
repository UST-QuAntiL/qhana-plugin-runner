# Optimization Plugin Interaction (coarse)

Below you will find a coarse sequence diagram of the plugin-to-plugin interactions between
the coordinator plugin and the [](../objective-function.md) and [](../minimizer.md) that the
user chose for the optimization.

```{mermaid}
sequenceDiagram
    actor User
    participant Coord
    participant Mini
    participant OF

    User->>+Coord: Select Coord Plugin
    Coord->>-User: Coord Setup UI
    
    User->>+Coord: Input
    
    Coord->>+OF: Start setup
    OF->>User: OF Setup UI
    User->>OF: Input
    OF->>-Coord: Setup finished
    
    Coord->>+Mini: Start setup
    Mini->>User: Setup UI
    User->>Mini: Input
    Mini->>-Coord: Setup finished
    
    Coord->>+Mini: Start minimization
    loop
        Mini->>+OF: Calc OF
        OF->>-Mini: Loss
    end
    Mini->>-Coord: Minimization finished
    
    Coord->>-User: Finished
```
