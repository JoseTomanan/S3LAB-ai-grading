<script lang="ts">
  let {
    checked = $bindable(false),
    id,
    disabled = false,
    ...restProps
  }: {
    checked?: boolean;
    id?: string;
    disabled?: boolean;
    [key: string]: unknown;
  } = $props();
</script>

<button
  {id}
  type="button"
  role="switch"
  aria-checked={checked}
  {disabled}
  onclick={() => { if (!disabled) checked = !checked; }}
  class="relative inline-flex shrink-0 items-center rounded-full border-2 border-transparent
         transition-colors duration-200 cursor-pointer
         focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2
         disabled:cursor-not-allowed disabled:opacity-50
         h-5 w-9"
  class:bg-primary={checked}
  class:bg-border={!checked}
  {...restProps}
>
  <span
    class="pointer-events-none block rounded-full bg-background shadow transition-transform duration-200
           size-4"
    class:translate-x-4={checked}
    class:translate-x-0={!checked}
  ></span>
</button>
