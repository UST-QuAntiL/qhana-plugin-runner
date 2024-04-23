# Optimization Plugin Interaction (detailed)

Below you will find a detailed sequence diagram of the plugin-to-plugin interactions between
the coordinator plugin and the [](../objective-function.md) and [](../minimizer.md) that the
user chose for the optimization.

```{mermaid}
sequenceDiagram
    actor User
    participant Coord
    participant OF
    participant Minimizer
    %% TODO: add every add_step task

    %%User->>Coord: get /ui-setup/
    %%Coord->>User: Coord Setup UI
    Note over User,Coord: Coord Setup UI
    
    User->>Coord: post setup
    Coord->>Coord: start add_plugin_entrypoint_task
    Coord->>User: task URL
    %%Coord->>User: new step: OF setup
    
    %%User->>OF: get /ui-hyperparameter/?callback=...
    %%OF->>User: OF Hyperparameter UI
    Note over User, OF: OF Setup UI
    
    User->>OF: post hyperparameters (callback=...)
    
    loop stepID != "pass_data"
        OF--)Coord: call OF webhook (event=steps)
        Coord->>Coord: start check_of_steps
    end
    
    Coord->>OF: post data (stepID == "pass_data")
    OF->>OF: start load_data
    
    loop stepID != "evaluate"
        OF--)Coord: call OF webhook (event=steps)
        Coord->>Coord: start check_of_steps
    end
    
    Note over User,Minimizer: Minimizer Setup UI
    
    User->>Minimizer: post setup
    
    loop stepID != "minimize"
        Minimizer--)Coord: call minimizer webhook (event=steps)
        Coord->>Coord: start check_minimizer_steps
    end
    
    Note over Coord,Minimizer: start minimization
    
    Coord->>Minimizer: post minimize
    
    Minimizer->>OF: get "of-weights"
    OF->>Minimizer: return number of weights

    loop converged
        Minimizer->>OF: post "of-evaluate"
        OF->>Minimizer: return loss
    end
    
    Note over Coord,Minimizer: start cleanup
    
    loop status == "PENDING"
        Minimizer--)Coord: call minimizer webhook (event=steps)
        Coord->>Coord: start check_minimizer_steps
    end
    
    Minimizer--)Coord: call minimizer webhook (event=status)
    
    loop status == "PENDING"
        OF--)Coord: call OF webhook (event=steps)
        Coord->>Coord: start check_of_steps
    end
    
    OF--)Coord: call OF webhook (event=status)
    
    Coord->>User: present result
```
