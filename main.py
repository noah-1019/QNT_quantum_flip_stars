import numpy as np

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
        New Fisher Information matrix [[f_00,f_01,f_02],
                                        [f_10,f_11,f_12], 
                                        [f_20,f_21,f_22]] containing the calculated contributions 
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
    - The fisher information matrix is symmetric so only the upper triangle is filled
    - The function raises a ZeroDivisionError if any theta value equals 0 or 1 in protocols that use 1/(θ-θ²)
    Raises:
    -------
    ZeroDivisionError
        If any theta value equals 0 or 1 in protocols that use 1/(θ-θ²) which is not physically possible.
    """
  
    # Initialize Fisher Information list if not provided
    fisher_info = np.zeros((len(theta), len(theta)), dtype=float)


    # ----- Multi cast Protocol -----
    if protocol == 0:

        theta_0 = theta[root]
        theta_1 = theta[(root + 1) % len(theta)]
        theta_2 = theta[(root + 2) % len(theta)]

      
        # -------------------------------------------------------------------#
        ## Calculate Diagonals ##
        # -------------------------------------------------------------------#

        # Root channel information
        fisher_info[root,root] += (
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
        fisher_info[(root + 1) % len(fisher_info), (root + 1) % len(fisher_info)] += (
        (theta_0 - theta_2)**2 / 
        (theta_0 * theta_1 + theta_2 - theta_0 * theta_2 - theta_1 * theta_2) + 

        (1 - theta_0 - theta_2)**2 / 
        (theta_1 - theta_0 * theta_1 + theta_0 * theta_2 - theta_1 * theta_2) + 

        (-theta_0 + theta_2)**2 / 
        (theta_0 - theta_0 * theta_1 - theta_0 * theta_2 + theta_1 * theta_2) + 

        (-1 + theta_0 + theta_2)**2 / 
        (1 - theta_0 - theta_1 + theta_0 * theta_1 - theta_2 + theta_0 * theta_2 + theta_1 * theta_2)
        )

        fisher_info[(root + 2) % len(fisher_info), (root + 2) % len(fisher_info)] += (

        (1 - theta_0 - theta_1)**2 / 
        (theta_0 * theta_1 + theta_2 - theta_0 * theta_2 - theta_1 * theta_2) + 

        (theta_0 - theta_1)**2 / 
        (theta_1 - theta_0 * theta_1 + theta_0 * theta_2 - theta_1 * theta_2) + 

        (-theta_0 + theta_1)**2 / 
        (theta_0 - theta_0 * theta_1 - theta_0 * theta_2 + theta_1 * theta_2) + 

        (-1 + theta_0 + theta_1)**2 / 
        (1 - theta_0 - theta_1 + theta_0 * theta_1 - theta_2 + theta_0 * theta_2 + theta_1 * theta_2)

        )

        # -------------------------------------------------------------------#
        ## Calculate Off-Diagonals ##
        # -------------------------------------------------------------------#

        # F_01 (Assuming root is 0)
        fisher_info[root, (root + 1) % len(fisher_info)] += (
          ((theta_0 - theta_2) * (theta_1 - theta_2)) / 
          (theta_0 * theta_1 + theta_2 - theta_0 * theta_2 - theta_1 * theta_2) + 

          ((1 - theta_0 - theta_2) * (-theta_1 + theta_2)) / 
          (theta_1 - theta_0 * theta_1 + theta_0 * theta_2 - theta_1 * theta_2) + 

          ((1 - theta_1 - theta_2) * (-theta_0 + theta_2)) / 
          (theta_0 - theta_0 * theta_1 - theta_0 * theta_2 + theta_1 * theta_2) + 

          ((-1 + theta_0 + theta_2) * (-1 + theta_1 + theta_2)) / 
          (1 - theta_0 - theta_1 + theta_0 * theta_1 - theta_2 + theta_0 * theta_2 + theta_1 * theta_2)
        )   

        # F_02 (Assuming root is 0)
        fisher_info[root, (root + 2) % len(fisher_info)] += (
            ((1 - theta_0 - theta_1) * (theta_1 - theta_2)) / 
            (theta_0 * theta_1 + theta_2 - theta_0 * theta_2 - theta_1 * theta_2) + 

            ((theta_0 - theta_1) * (-theta_1 + theta_2)) / 
            (theta_1 - theta_0 * theta_1 + theta_0 * theta_2 - theta_1 * theta_2) + 

            ((-theta_0 + theta_1) * (1 - theta_1 - theta_2)) / 
            (theta_0 - theta_0 * theta_1 - theta_0 * theta_2 + theta_1 * theta_2) + 

            ((-1 + theta_0 + theta_1) * (-1 + theta_1 + theta_2)) / 
            (1 - theta_0 - theta_1 + theta_0 * theta_1 - theta_2 + theta_0 * theta_2 + theta_1 * theta_2)
          )

        # F_12 (Assuming root is 0)
        fisher_info[(root + 1) % len(fisher_info), (root + 2) % len(fisher_info)] += (
            ((1 - theta_0 - theta_1) * (theta_0 - theta_2)) / 
            (theta_0 * theta_1 + theta_2 - theta_0 * theta_2 - theta_1 * theta_2) + 

            ((theta_0 - theta_1) * (1 - theta_0 - theta_2)) / 
            (theta_1 - theta_0 * theta_1 + theta_0 * theta_2 - theta_1 * theta_2) + 

            ((-theta_0 + theta_1) * (-theta_0 + theta_2)) / 
            (theta_0 - theta_0 * theta_1 - theta_0 * theta_2 + theta_1 * theta_2) + 

            ((-1 + theta_0 + theta_1) * (-1 + theta_0 + theta_2)) / 
            (1 - theta_0 - theta_1 + theta_0 * theta_1 - theta_2 + theta_0 * theta_2 + theta_1 * theta_2)
          )

    # ----- Root Independent Protocol -----
    # In this protocol, the parameter nearest to the root does not receive any information. The other parameters receive information based on their own value. All off diagonal elements are zero.
    elif protocol == 1: # Root Independent
        for i in range(len(theta)): # Loop through all parameters
            if i != root: 
                fisher_info[i,i]+=1/(theta[i]-theta[i]**2)



    # ----- Independent Encoding Protocol X basis -----
    # In this protocol, the root parameter receives information based on its own value, while the other parameters dont receive an information. All off diagonal elements are zero.
    elif protocol == 2: # Independent Encoding X basis
        fisher_info[root, root] += 1/(theta[root]-theta[root]**2) # The root parameter receives information based on its own value.


    # ----- Independent Encoding Protocol Z basis -----
    # In this protocol, all parameters receive information based on their own value. Very similarly to the Root Independent Protocol, but without the root parameter. All off diagonal elements are zero.
    elif protocol == 3: # Independent Encoding Z basis
        for i in range(len(fisher_info)): # Loop through all parameters
            fisher_info[i,i]+=1/(theta[i]-theta[i]**2)



    # ----- Back and Forth Protocol Z basis -----
    # This protocol is complex similar to the multicast protocol; however, this protocol uses 3 qubits instead of 2.
    elif protocol == 4:
        theta_0 = theta[root]
        theta_1 = theta[(root + 1) % len(theta)]
        theta_2 = theta[(root + 2) % len(theta)]

        # -------------------------------------------------------------------#
        ## Calculate Diagonals ##
        # -------------------------------------------------------------------#

        # Root channel information
        fisher_info[root,root] += (
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
        fisher_info[(root + 1) % len(fisher_info), (root + 1) % len(fisher_info)] += (
        (theta_0 - theta_2)**2 / 
        (theta_0 * theta_1 + theta_2 - theta_0 * theta_2 - theta_1 * theta_2) + 

        (1 - theta_0 - theta_2)**2 / 
        (theta_1 - theta_0 * theta_1 + theta_0 * theta_2 - theta_1 * theta_2) + 

        (-theta_0 + theta_2)**2 / 
        (theta_0 - theta_0 * theta_1 - theta_0 * theta_2 + theta_1 * theta_2) + 

        (-1 + theta_0 + theta_2)**2 / 
        (1 - theta_0 - theta_1 + theta_0 * theta_1 - theta_2 + theta_0 * theta_2 + theta_1 * theta_2)
        )

        fisher_info[(root + 2) % len(fisher_info), (root + 2) % len(fisher_info)] += (
        (1 - theta_0 - theta_1)**2 / 
        (theta_0 * theta_1 + theta_2 - theta_0 * theta_2 - theta_1 * theta_2) + 
    
        (theta_0 - theta_1)**2 / 
        (theta_1 - theta_0 * theta_1 + theta_0 * theta_2 - theta_1 * theta_2) + 
    
        (-theta_0 + theta_1)**2 / 
        (theta_0 - theta_0 * theta_1 - theta_0 * theta_2 + theta_1 * theta_2) + 
    
        (-1 + theta_0 + theta_1)**2 / 
        (1 - theta_0 - theta_1 + theta_0 * theta_1 - theta_2 + theta_0 * theta_2 + theta_1 * theta_2)
        )

        # -------------------------------------------------------------------#
        ## Calculate Off- Diagonals ##
        # -------------------------------------------------------------------#

        # F_01 (Assuming root is 0)
        fisher_info[root, (root + 1) % len(fisher_info)] += (
            ((theta_0 - theta_2) * (theta_1 - theta_2)) / 
            (theta_0 * theta_1 + theta_2 - theta_0 * theta_2 - theta_1 * theta_2) + 

            ((1 - theta_0 - theta_2) * (-theta_1 + theta_2)) / 
            (theta_1 - theta_0 * theta_1 + theta_0 * theta_2 - theta_1 * theta_2) + 

            ((1 - theta_1 - theta_2) * (-theta_0 + theta_2)) / 
            (theta_0 - theta_0 * theta_1 - theta_0 * theta_2 + theta_1 * theta_2) + 

            ((-1 + theta_0 + theta_2) * (-1 + theta_1 + theta_2)) / 
            (1 - theta_0 - theta_1 + theta_0 * theta_1 - theta_2 + theta_0 * theta_2 + theta_1 * theta_2)
        )

        # F_02 (Assuming root is 0)
        fisher_info[root, (root + 2) % len(fisher_info)] += (
            ((1 - theta_0 - theta_1) * (theta_1 - theta_2)) / 
            (theta_0 * theta_1 + theta_2 - theta_0 * theta_2 - theta_1 * theta_2) + 

            ((theta_0 - theta_1) * (-theta_1 + theta_2)) / 
            (theta_1 - theta_0 * theta_1 + theta_0 * theta_2 - theta_1 * theta_2) + 

            ((-theta_0 + theta_1) * (1 - theta_1 - theta_2)) / 
            (theta_0 - theta_0 * theta_1 - theta_0 * theta_2 + theta_1 * theta_2) + 

            ((-1 + theta_0 + theta_1) * (-1 + theta_1 + theta_2)) / 
            (1 - theta_0 - theta_1 + theta_0 * theta_1 - theta_2 + theta_0 * theta_2 + theta_1 * theta_2)
        )
        
        # F_12 (Assuming root is 0)
        fisher_info[(root + 1) % len(fisher_info), (root + 2) % len(fisher_info)] += (
            ((1 - theta_0 - theta_1) * (theta_0 - theta_2)) / 
            (theta_0 * theta_1 + theta_2 - theta_0 * theta_2 - theta_1 * theta_2) + 

            ((theta_0 - theta_1) * (1 - theta_0 - theta_2)) / 
            (theta_1 - theta_0 * theta_1 + theta_0 * theta_2 - theta_1 * theta_2) + 

            ((-theta_0 + theta_1) * (-theta_0 + theta_2)) / 
            (theta_0 - theta_0 * theta_1 - theta_0 * theta_2 + theta_1 * theta_2) + 

            ((-1 + theta_0 + theta_1) * (-1 + theta_0 + theta_2)) / 
            (1 - theta_0 - theta_1 + theta_0 * theta_1 - theta_2 + theta_0 * theta_2 + theta_1 * theta_2)
        )


    # ----- Back and Forth Protocol X basis -----
    # In this protocol, the root parameter receives information based on its own value, while the other parameters dont receive an information. This is exactly the same as the Independent Encoding Protocol X basis.
    elif protocol == 5:
        fisher_info[root, root] += 1/(theta[root]-theta[root]**2) # The root parameter receives information based on its own value.


    # Fill the lower triangle of the matrix to maintain symmetry
    for i in range(len(fisher_info)):
        for j in range(i + 1, len(fisher_info)):
            if fisher_info[i, j] != 0:
                # Copy upper triangle to lower triangle
                fisher_info[j, i] = fisher_info[i, j]
            else:
              fisher_info[i, j] = fisher_info[j, i]

    return fisher_info

def reward_function(fisher_matrix):
    """
    Calculates the reward based on the Fisher Information matrix.
    This function computes a reward value that is inversely proportional to the trace 
    of the inverse Fisher Information matrix. The trace of the inverse Fisher matrix
    represents the sum of the Cramér-Rao lower bounds for parameter estimation,
    so minimizing this trace improves overall parameter estimation precision.
    
    Parameters:
    -----------
    fisher_matrix : numpy.ndarray
        The Fisher Information matrix (3x3) containing information contributions 
        for each parameter and their correlations
    
    Returns:
    --------
    float
        The calculated reward value inversely proportional to tr(F⁻¹).
        Higher Fisher information (lower estimation uncertainty) gives higher reward.
        
    Notes:
    ------
    - The trace of the inverse Fisher matrix equals the sum of variances in optimal estimation
    - Lower trace means better parameter estimation precision
    - Reward = 1 / (trace(F⁻¹) + ε) where ε prevents division by zero
    - If matrix is singular (non-invertible), returns a small penalty reward
    """
    
    try:
        # Compute the inverse of the Fisher Information matrix
        fisher_inverse = np.linalg.inv(fisher_matrix)
        
        # Calculate the trace of the inverse matrix
        trace_inverse = np.trace(fisher_inverse)
        
        # Small epsilon to prevent division by zero and ensure numerical stability
        epsilon = 1e-8
        
        # Reward is inversely proportional to the trace of the inverse
        # Higher Fisher information (lower trace) gives higher reward
        reward = 1.0 / (trace_inverse + epsilon)
        
        
        return np.log(reward + 1)
        
    except np.linalg.LinAlgError:
        # Matrix is singular (non-invertible), create submatrix by removing zero diagonal elements
        # This happens when Fisher matrix has zero determinant (e.g., some parameters have zero info)
        try:
            # Find indices of diagonal elements that are non-zero (above tolerance)
            tolerance = 1e-10
            diagonal = np.diag(fisher_matrix)
            non_zero_indices = np.where(np.abs(diagonal) > tolerance)[0]
            
            if len(non_zero_indices) == 0:
                # All diagonal elements are zero, return penalty
                return -1e-6
            
            # Create submatrix by selecting only rows and columns with non-zero diagonal elements
            submatrix = fisher_matrix[np.ix_(non_zero_indices, non_zero_indices)]
            
            # Compute inverse of the reduced submatrix
            submatrix_inverse = np.linalg.inv(submatrix)
            
            # Calculate trace of the submatrix inverse
            trace_inverse = np.trace(submatrix_inverse)
            
            # Small epsilon to prevent division by zero
            epsilon = 1e-8
            
            # Reward based on reduced submatrix (partial parameter estimation)
            reward = 1.0 / (trace_inverse + epsilon)
            
            return np.log(reward + 1)
            
        except np.linalg.LinAlgError:
            # Even reduced submatrix is singular, return small penalty reward
            return -1e-6
