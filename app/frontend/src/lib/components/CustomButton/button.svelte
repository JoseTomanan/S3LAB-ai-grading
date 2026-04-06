<script lang="ts" module>
	import { cn, type WithElementRef } from "$lib/utils.js";
	import type { HTMLAnchorAttributes, HTMLButtonAttributes } from "svelte/elements";
	import { type VariantProps, tv } from "tailwind-variants";

	export const buttonVariants = tv({
		base: "inline-flex shrink-0 items-center justify-center gap-x-2 rounded text-sm font-medium cursor-pointer transition-all outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] disabled:pointer-events-none disabled:opacity-50 aria-disabled:pointer-events-none aria-disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
		variants: {
			size: {
				default: "px-2 py-1",
				sm: "px-1.5 py-0.5 gap-x-1.5",
				lg: "px-4 py-2",
				icon: "size-9",
				"icon-sm": "size-8",
				"icon-lg": "size-10",
			},
			variant: {
				default: "bg-primary text-primary-foreground shadow-xs hover:bg-primary/90",
				secondary: "bg-secondary text-secondary-foreground shadow-xs hover:bg-secondary/80",
				destructive: "bg-destructive text-white shadow-xs hover:opacity-80",
				outline: "bg-card text-card-foreground shadow-xs ring ring-border border-0",
				ghost: "hover:bg-accent hover:text-accent-foreground",
				link: "text-primary underline-offset-4 hover:underline",
				floating:
					"bg-white hover:from-neutral-50 hover:to-neutral-200 text-[var(--color-primary-700)] border border-border shadow rounded-full h-fit p-0",
				"floating-secondary":
					"bg-white hover:from-neutral-50 hover:to-neutral-200 text-[var(--secondary-700)] border border-border shadow rounded-full h-fit p-0",
			},
		},
		defaultVariants: {
			variant: "default",
			size: "default",
		},
	});

	export type ButtonVariant = VariantProps<typeof buttonVariants>["variant"];
	export type ButtonSize = VariantProps<typeof buttonVariants>["size"];

	export type ButtonProps = WithElementRef<HTMLButtonAttributes> &
		WithElementRef<HTMLAnchorAttributes> & {
			variant?: ButtonVariant;
			size?: ButtonSize;
		};
</script>

<script lang="ts">
	let {
		class: className,
		variant = "default",
		size = "default",
		ref = $bindable(null),
		href = undefined,
		type = "button",
		disabled,
		children,
		...restProps
	}: ButtonProps = $props();
</script>

{#if href}
	<a
		bind:this={ref}
		data-slot="button"
		class={cn(buttonVariants({ variant, size }), className)}
		href={disabled ? undefined : href}
		aria-disabled={disabled}
		role={disabled ? "link" : undefined}
		tabindex={disabled ? -1 : undefined}
		{...restProps}
	>
		{@render children?.()}
	</a>
{:else}
	<button
		bind:this={ref}
		data-slot="button"
		class={cn(buttonVariants({ variant, size }), className)}
		{type}
		{disabled}
		{...restProps}
	>
		{@render children?.()}
	</button>
{/if}
