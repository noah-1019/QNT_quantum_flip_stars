def updated_fisher_information(root, protocol, theta):
    """
    Calculates the Fisher Information matrix for quantum network parameters based on the chosen protocol.
    
    This function computes Fisher Information contributions for each parameter depending on 
    the quantum measurement protocol being used in a 3-node quantum flip star network.
    
    Parameters:
    -----------
    root : int
        The index of the root node/channel (0, 1, or 2) that serves as the central measurement point
    protocol : int
        The quantum protocol to use:
        - 0: Multicast Protocol - All channels receive complex Fisher information updates
        - 1: Root Independent Protocol - All channels except root receive 1/(θ-θ²) information
        - 2: Independent Encoding X basis - Only root receives 1/(θ-θ²) information  
        - 3: Independent Encoding Z basis - All channels receive 1/(θ-θ²) information
        - 4: Back and Forth Z basis - Complex Fisher information updates for all channels
        - 5: Back and Forth X basis - Only root receives 1/(θ-θ²) information
    theta : list
        Quantum channel error parameters [θ₀, θ₁, θ₂] representing flip probabilities
        Must be in range (0,1) and not equal to 0 or 1 to avoid division by zero
        
    Returns:
    --------
    list
        New Fisher Information matrix [F₀, F₁, F₂] containing the calculated contributions
        for each parameter based on the specified protocol
        
    Notes:
    ------
    - Fisher Information quantifies how much information a measurement provides about parameters
    - Higher Fisher Information indicates better parameter estimation capability
    - The function uses modular arithmetic to handle circular indexing of the 3-node network
    - Protocols 0 and 4 use complex multivariate Fisher information formulas derived from 
      quantum network tomography theory
    - Protocols 1, 2, 3, 5 use the simpler univariate formula 1/(θᵢ(1-θᵢ))
    - The function initializes a new Fisher Information list with zeros for each call
    
    Raises:
    -------
    ZeroDivisionError
        If any theta value equals 0 or 1 in protocols that use 1/(θ-θ²) which is not physically possible.
    """
  
    # Initialize Fisher Information list if not provided
    fisher_info = [0.0] * len(theta) 
    # ----- Multi cast Protocol -----
    if protocol == 0:

        theta_0 = theta[root]
        theta_1 = theta[(root + 1) % len(theta)]
        theta_2 = theta[(root + 2) % len(theta)]

        #print(f"Root: {root}, Theta values: {theta_0}, {theta_1}, {theta_2}")

        # Root channel information
        fisher_info[root] += (
        (theta_1 - theta_2)**2 / 
        (theta_0 * theta_1 + theta_2 - theta_0 * theta_2 - theta_1 * theta_2) + 

        (-theta_1 + theta_2)**2 / 
        (theta_1 - theta_0 * theta_1 + theta_0 * theta_2 - theta_1 * theta_2) + 

        (1 - theta_1 - theta_2)**2 / 
        (theta_0 - theta_0 * theta_1 - theta_0 * theta_2 + theta_1 * theta_2) + 

        (-1 + theta_1 + theta_2)**2 / 
        (1 - theta_0 - theta_1 + theta_0 * theta_1 - theta_2 + theta_0 * theta_2 + theta_1 * theta_2)
        )
        #print(f"Fisher info for root {root}: {fisher_info[root]}")

        # Other Channel information
        fisher_info[(root + 1) % len(fisher_info)] += (
        (theta_0 - theta_2)**2 / 
        (theta_0 * theta_1 + theta_2 - theta_0 * theta_2 - theta_1 * theta_2) + 

        (1 - theta_0 - theta_2)**2 / 
        (theta_1 - theta_0 * theta_1 + theta_0 * theta_2 - theta_1 * theta_2) + 

        (-theta_0 + theta_2)**2 / 
        (theta_0 - theta_0 * theta_1 - theta_0 * theta_2 + theta_1 * theta_2) + 

        (-1 + theta_0 + theta_2)**2 / 
        (1 - theta_0 - theta_1 + theta_0 * theta_1 - theta_2 + theta_0 * theta_2 + theta_1 * theta_2)
        )

        fisher_info[(root + 2) % len(fisher_info)] += (
        (1 - theta_0 - theta_1)**2 / 
        (theta_0 * theta_1 + theta_2 - theta_0 * theta_2 - theta_1 * theta_2) + 

        (theta_0 - theta_1)**2 / 
        (theta_1 - theta_0 * theta_1 + theta_0 * theta_2 - theta_1 * theta_2) + 

        (-theta_0 + theta_1)**2 / 
        (theta_0 - theta_0 * theta_1 - theta_0 * theta_2 + theta_1 * theta_2) + 

        (-1 + theta_0 + theta_1)**2 / 
        (1 - theta_0 - theta_1 + theta_0 * theta_1 - theta_2 + theta_0 * theta_2 + theta_1 * theta_2)
        )


    # ----- Root Independent Protocol -----
    # In this protocol, the parameter nearest to the root does not receive any information. The other parameters receive information based on their own value.
    elif protocol == 1: # Root Independent
        for i in range(len(fisher_info)): # Loop through all parameters
            if i != root: 
                fisher_info[i]+=1/(theta[i]-theta[i]**2)

    # ----- Independent Encoding Protocol X basis -----
    # In this protocol, the root parameter receives information based on its own value, while the other parameters dont receive an information.
    elif protocol == 2: # Independent Encoding X basis
        fisher_info[root]+= 1/(theta[root]-theta[root]**2) # The root parameter receives information based on its own value.

    # ----- Independent Encoding Protocol Z basis -----
    # In this protocol, all parameters receive information based on their own value. Very similarly to the Root Independent Protocol, but without the root parameter.
    elif protocol == 3: # Independent Encoding Z basis
        for i in range(len(fisher_info)): # Loop through all parameters
            fisher_info[i]+=1/(theta[i]-theta[i]**2)

    # ----- Back and Forth Protocol Z basis -----
    elif protocol == 4: 
        theta_0 = theta[root]
        theta_1 = theta[(root + 1) % len(theta)]
        theta_2 = theta[(root + 2) % len(theta)]

        # Root channel information
        fisher_info[root] += (
        -((theta_1 - theta_2)**2 / 
          (-theta_0 * theta_2 + theta_1 * (-1 + theta_0 + theta_2)))    

        - (theta_1 - theta_2)**2 / 
          ((-1 + theta_1) * theta_2 + theta_0 * (-theta_1 + theta_2)) 

        + (-1 + theta_1 + theta_2)**2 / 
          (theta_1 * theta_2 - theta_0 * (-1 + theta_1 + theta_2)) 

        + (-1 + theta_1 + theta_2)**2 / 
          ((-1 + theta_1) * (-1 + theta_2) + theta_0 * (-1 + theta_1 + theta_2))
        ) 

        # Other channel information
        fisher_info[(root + 1) % len(fisher_info)] += (
        (theta_0 - theta_2)**2 / 
        (theta_0 * theta_1 + theta_2 - theta_0 * theta_2 - theta_1 * theta_2) + 

        (1 - theta_0 - theta_2)**2 / 
        (theta_1 - theta_0 * theta_1 + theta_0 * theta_2 - theta_1 * theta_2) + 

        (-theta_0 + theta_2)**2 / 
        (theta_0 - theta_0 * theta_1 - theta_0 * theta_2 + theta_1 * theta_2) + 

        (-1 + theta_0 + theta_2)**2 / 
        (1 - theta_0 - theta_1 + theta_0 * theta_1 - theta_2 + theta_0 * theta_2 + theta_1 * theta_2)
        )

        fisher_info[(root + 2) % len(fisher_info)] += (
        (1 - theta_0 - theta_1)**2 / 
        (theta_0 * theta_1 + theta_2 - theta_0 * theta_2 - theta_1 * theta_2) + 
    
        (theta_0 - theta_1)**2 / 
        (theta_1 - theta_0 * theta_1 + theta_0 * theta_2 - theta_1 * theta_2) + 
    
        (-theta_0 + theta_1)**2 / 
        (theta_0 - theta_0 * theta_1 - theta_0 * theta_2 + theta_1 * theta_2) + 
    
        (-1 + theta_0 + theta_1)**2 / 
        (1 - theta_0 - theta_1 + theta_0 * theta_1 - theta_2 + theta_0 * theta_2 + theta_1 * theta_2)
        )

    # ----- Back and Forth Protocol X basis -----
    # In this protocol, the root parameter receives information based on its own value, while the other parameters dont receive an information. This is exactly the same as the Independent Encoding Protocol X basis.
    elif protocol == 5:
        fisher_info[root]+= 1/(theta[root]-theta[root]**2) # The root parameter receives information based on its own value.

    return fisher_info

def reward_function(old_fisher_info,new_fisher_info):
    """
    Calculates the reward based on the Fisher Information and theta values.
    This function computes a reward value based on the Fisher Information matrix diagonal values
    from the previous and current states. This reward function is designed to encourage
    exploration of quantum network parameters by balancing the mean Fisher Information
    against the variance of the Fisher Information contributions.
    
    Parameters:
    -----------
    fisher_info : list
        The Fisher Information matrix diagonal values[ F₀, F₁, F₂] containing contributions for each parameter
    
    Returns:
    --------
    float
        The calculated reward value based on the Fisher Information.
    """

    p= .05  # Example penalty factor, adjust as needed

    mean_info=sum(new_fisher_info)/len(new_fisher_info)

    
    # Element-wise addition of the two lists
    combined_info = [old + new for old, new in zip(old_fisher_info, new_fisher_info)]
    total_mean = sum(combined_info) / len(combined_info)
    
    # Calculate the variance among the values in the fisher information matrix diagonal
    variance = sum((x - total_mean) ** 2 for x in combined_info) / len(combined_info)

    # Add your reward calculation logic here
    # For example:
    reward = mean_info - p * variance  # Just an example
    
    return reward


