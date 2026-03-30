<script lang="ts">
  import type { Snippet } from 'svelte';
  import type { HTMLAnchorAttributes, HTMLAttributes } from 'svelte/elements';

  type BaseProps = {
    children: Snippet;
    class?: string;
  };

  type AnchorProps = BaseProps & HTMLAnchorAttributes & { href: string };
  type DivProps = BaseProps & HTMLAttributes<HTMLDivElement> & { href?: never };

  type Props = AnchorProps | DivProps;

  let { children, class: className = "", href, ...rest }: Props = $props();

  let tag = $derived(href ? 'a' : 'div');
</script>


<svelte:element this={tag} 
        {...(href ? { href } : {})}
        class="bg-card text-card-foreground rounded-xs border border-border/50 px-3 py-1.5 {className}" 
        {...rest}>
  {@render children()}
</svelte:element>
