<script lang="ts">
  import type { Snippet } from 'svelte';
  import type { HTMLAnchorAttributes, HTMLAttributes } from 'svelte/elements';
  import { cn } from '$lib/utils.ts';

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
        class={cn("bg-card text-card-foreground border-2 border-border/60 shadow-none px-3 py-1.5", className)}
        {...rest}>
  {@render children()}
</svelte:element>
