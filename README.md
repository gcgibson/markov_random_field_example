<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>


### Undirected Graphical Models
I have gotten pretty into ugms lately because of their application to biostats. Bayesian nets play a prominent role in cancer diagnosis, and with the advances in deep learning, I think that undirected graphical models might mean big things for diagnostic tools. The main advantage of ugms versus Bayesian networks is that they don't require any domain knowledge of causality. However, they still have a couple of problems,
1. Intractable partition function for large networks 
2. Intractable learning graph structure from data (partially due to 1.)

These two issues *might* soon be remedied by advances in deep learning. 

For now we need to get a handle on Markov Random Fields or undirected graphical models, or whatever you want to call them. A lot of tutorials start quickly and get deep into the math without going over the basics so I want to take a look a the following example.


![optional caption text](figures/img.png)

$$Val(X_1) = \{{1,2}\}$$ 
$$Val(X_2) = \{{1,2}\}$$
That is both $$X_1$$ and $$X_2$$ can either be on ($$1$$) or off ($$2$$).
We can see that $$X_1 - X_2$$ implies that that $$P(X_1,X_2) \ \ != P(X_1)*P(X_2)$$.
If we look at the definition of a Markov random field we have:

$$P(X_1,X_2) = \frac{1}{Z}\phi_{X_1}(X_1)\phi_{X_2}(X_2)\phi_{X_1,X_2}(X_1,X_2)$$

This notation is a little tricky but all we are really saying is take a a function $$\phi_{X_1}$$ defined over $$X_1$$ that spits out a number. Let's call this a factor. 
This function $$\phi$$ is defined for *both nodes and edges* and the probability of the joint $$P(X_1,X_2)$$ is just the product of the *factors* of the graph. In the binary case that is simply the expression above. We call $$\phi$$ a potential, which I guess has something to do with physics, but just think of it as an unnormalized affinity to be in a particular state.

The literature on MRFs makes things even more confusing by jumping straight into energy functions and examining the ISING model. Forget about that for now and just focus on how potentials are multiplied together.

But wait! What the heck is that $$Z$$? 
That is the infamous partition function, which is just a way of normalizing the product of potentials to be in $$[0,1]$$ so we can call it a probability distribution. It is simply the product of potentials over all possible values of $$X_1$$ and $$X_2$$. We can write it as follows

$$Z = \sum_{Val(v1 \in X_1)}\sum_{v_2 \in 
Val(X_2)}\phi_{X_1}(v_1)\phi_{X_2}(v_2)\phi_{X_1,X_2}(v_1,v_2)$$

Stare at that for a second and it will make sense.

I will add more here later, but for now you can play around with **two_node_mrf.py** to get a sense of how we learn MRFs from data. To start with I included a very simple example of two binary valued random variables $$X_1,X_2$$ that are perfectly correlated. This means that when
$$X_1 = 1 \leftrightarrow X_2 = 1$$  
$$X_1 = 2 \leftrightarrow X_2 = 2$$  

Enjoy!

